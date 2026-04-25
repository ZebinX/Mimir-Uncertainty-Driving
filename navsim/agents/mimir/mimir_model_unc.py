from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
from navsim.agents.mimir.mimir_config import MimirConfig
from navsim.agents.mimir.mimir_backbone import MimirBackbone
from navsim.agents.mimir.mimir_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from diffusers.schedulers import DDIMScheduler
from navsim.agents.mimir.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
import torch.nn.functional as F
from navsim.agents.mimir.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention,GridSampleCrossBEVAttention_navi,GridSampleCrossBEVAttention_naviunc
from navsim.agents.mimir.modules.multimodal_loss import LossComputer,LaplaceNLLLoss
from torch.nn import TransformerDecoder,TransformerDecoderLayer
from typing import Any, List, Dict, Optional, Union,Tuple
import numpy.typing as npt
import numpy as np
from navsim.agents.mimir.modules.blocks import init_weights


def _transform_navi_to_camera_tensor(
    navi: torch.Tensor,                     # (N, 3)
    sensor2lidar_rotation: torch.Tensor,   # (3, 3)
    sensor2lidar_translation: torch.Tensor # (3,)
) -> torch.Tensor:
    """
    将导航点从 VCS 坐标变换到摄像机坐标系
    """
    lidar2cam_r = sensor2lidar_rotation.inverse()
    lidar2cam_t = -torch.matmul(lidar2cam_r, sensor2lidar_translation)

    locs_homo = torch.cat([navi, torch.ones_like(navi[:, :1])], dim=-1)  # (N, 4)

    # 构建 4x4 齐次变换矩阵
    lidar2cam_rt = torch.eye(4, device=navi.device).to(navi)
    lidar2cam_rt[:3, :3] = lidar2cam_r
    lidar2cam_rt[:3, 3] = lidar2cam_t

    locs_cam = torch.matmul(lidar2cam_rt, locs_homo.T).T[:, :3]  # (N, 3)
    return locs_cam

def _project_points_to_image_tensor(
    points: torch.Tensor,           # (N, 3) in camera frame
    intrinsics: torch.Tensor,      # (3, 3)
    image_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将摄像机系下的点投影到图像像素平面
    返回 (N, 2) 像素位置 + mask 是否在图像有效区域
    """
    N = points.shape[0]
    pc_homo = torch.cat([points, torch.ones((N, 1), device=points.device,dtype=points.dtype)], dim=-1)  # (N, 4)
    intrinsics_pad = torch.eye(4, device=points.device,dtype=points.dtype)
    intrinsics_pad[:3, :3] = intrinsics

    proj = torch.matmul(pc_homo, intrinsics_pad.T)  # (N, 4)
    z = proj[:, 2:3].clamp(min=eps)
    xy = proj[:, 0:2] / z  # (N, 2)

    in_fov = proj[:, 2] > eps

    if image_shape is not None:
        H, W = image_shape
        u, v = xy[:, 0], xy[:, 1]
        in_bounds = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        in_fov = in_fov & in_bounds

    return xy, in_fov

def extract_feature_values_at_navi_batched(
    feature_map: torch.Tensor,                 # (B, C_feat, H_feat, W_feat)
    navi_tensor: torch.Tensor,                 # (B, N, 3)
    sensor2lidar_rotation: torch.Tensor,       # (B, 3, 3)
    sensor2lidar_translation: torch.Tensor,    # (B, 3)
    intrinsics: torch.Tensor,                  # (B, 3, 3)
    image_shape: Tuple[int, int]               # 原始图像尺寸 (H_img, W_img)
) -> torch.Tensor:
    B, C_feat, H_feat, W_feat = feature_map.shape
    _, N, _ = navi_tensor.shape
    H_img, W_img = image_shape
    device = navi_tensor.device

    feature_values = torch.zeros((B, C_feat, N), device=device)

    for b in range(B):
        # 单图提取
        navi_cam = _transform_navi_to_camera_tensor(
            navi_tensor[b], sensor2lidar_rotation[b], sensor2lidar_translation[b]
        )
        pixel_coords, valid_mask = _project_points_to_image_tensor(
            navi_cam, intrinsics[b], image_shape=image_shape
        )
        # 缩放到特征图分辨率
        pixel_coords_scaled = pixel_coords.clone()
        pixel_coords_scaled[:, 0] *= W_feat / W_img
        pixel_coords_scaled[:, 1] *= H_feat / H_img

        u = pixel_coords_scaled[:, 0].round().long().clamp(0, W_feat - 1)
        v = pixel_coords_scaled[:, 1].round().long().clamp(0, H_feat - 1)

        feature_values[b] = feature_map[b, :, v, u] * valid_mask.unsqueeze(0)

    return feature_values  # shape: (B, C_feat, N)


class MimirModel(nn.Module):
    """Torch module for Mimir."""

    def __init__(self, config: MimirConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = MimirBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        if self._config.status_norm:
            self._status_encoding=nn.Linear(4+1+1,config.tf_d_model)
        else:
            self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)
        # 只用command
        # self._status_encoding = nn.Linear(4, config.tf_d_model)
        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=1,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1,320),
        )
        self.weight_score=None
        self.navi_bank=np.load(self._config.navi_bank_path,allow_pickle=True).item()
    
    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None,goalpoint=None,token=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        
        if self._config.latent:
            lidar_feature = None
        else:
            lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]
        
        if self._config.status_norm and self._config.training==False:
            vle = torch.norm(status_feature[:,4:6],dim=-1,keepdim=True)
            acc = torch.norm(status_feature[:,6:8],dim=-1,keepdim=True)
            dot_product = torch.sum(status_feature[:,4:6]*status_feature[:,6:8],dim=-1,keepdim=True)
            tag_vle = torch.where(status_feature[:,4:5] >= 0,1.0,-1.0)
            tag_acc = torch.where(dot_product > 0,1.0,-1.0)
            status_feature= torch.cat([status_feature[:,:4], tag_vle*vle,tag_acc*acc], dim=-1)

        if self._config.training:
            goalpoints=[]
            for t in features['token']:
                goalpoint = self.navi_bank[t]
                goalpoints.append(goalpoint)
            goalpoints=np.stack(goalpoints,axis=0)
            goalpoints=torch.from_numpy(goalpoints).to(camera_feature).unsqueeze(1) # (bs, 1, 2)
            goalpoint=goalpoints
            targets=targets['trajectory'][:,7:8,:2]
        else:
            goalpoints=[]
            goalpoint = self.navi_bank[token]
            goalpoints.append(goalpoint)
            goalpoints=np.stack(goalpoints,axis=0)
            goalpoints=torch.from_numpy(goalpoints).to(camera_feature).unsqueeze(1) # (bs, 1, 2)
            goalpoint=goalpoints
        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, img_feature = self._backbone(camera_feature, lidar_feature)
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        # upsample to the same shape as bev_feature_upscale

        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        # concat concat_cross_bev and cross_bev_feature
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))
        cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        trajectory = self._trajectory_head(trajectory_query,agents_query, cross_bev_feature,bev_spatial_shape,status_encoding[:, None],targets=targets,goalpoint=goalpoint)
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)
        return output
        # return output

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}


class MLPDecoder(nn.Module):

    def __init__(self,
                 embed_dims: int=256,
                 num_poses: int=1,
                 min_scale: float = 1e-3) -> None:
        super(MLPDecoder, self).__init__()
        self.embed_dims=embed_dims
        self.num_poses = num_poses
        self.min_scale = min_scale

        self.aggr_embed = nn.Sequential(
            nn.Linear(embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True))
        self.loc = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, 2))
        self.scale = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, 2))
        self.apply(init_weights)

    def forward(self,navis_feature) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.aggr_embed(navis_feature)
        loc = self.loc(out)
        scale = F.elu_(self.scale(out), alpha=1.0) + 1.0
        scale = scale + self.min_scale  # [F, N, H, 2]
        return loc,scale

class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                    global_cond, time_embed
                ], axis=-1)
        else:
            global_feature = time_embed
        # if global_img is not None:
        #     if len(global_img.shape)==4:
        #         global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
        #     else:
        #         global_img = global_img.permute(0,2,1).contiguous()
        #     global_feature = torch.cat([
        #             global_img, global_feature
        #         ], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale,shift = scale_shift.chunk(2,dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )

        # GridSampleCrossBEVAttention_navi
        self.cross_bev_attention_navi = GridSampleCrossBEVAttention_naviunc(
            config.tf_d_model,
            config.tf_num_head,
            num_points=1,
            config=config,
            in_bev_dims=256,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)

        self.use_proj_image=config.use_proj_image
        if config.use_proj_image:
            self.cross_imgnavi_attention = nn.MultiheadAttention(
                config.tf_d_model,
                config.tf_num_head,
                dropout=config.tf_dropout,
                batch_first=True,
            )
            self.dropout2 = nn.Dropout(0.1)
            self.norm4 = nn.LayerNorm(config.tf_d_model)
            self.time_modulation=ModulationLayer(config.tf_d_model,256)
        else:
            self.time_modulation = ModulationLayer(config.tf_d_model,256)
        self.task_decoder = MLPDecoder(
            embed_dims=config.tf_d_model,
            num_poses=1,
        )

    def forward(self, 
                navis_feature, 
                navis, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                status_encoding,
                ):
        # navis_feature (bs,1,256) navis (bs,1,2) bev_feature (4,256,64,64)
        navis_feature = self.cross_bev_attention_navi(navis_feature,navis,bev_feature,bev_spatial_shape)
        
        navis_feature = navis_feature + self.dropout(self.cross_agent_attention(navis_feature, agents_query,agents_query)[0])
        navis_feature = self.norm1(navis_feature)

        # 4.5 cross attention with  ego query
        navis_feature = navis_feature + self.dropout1(self.cross_ego_attention(navis_feature, ego_query,ego_query)[0])
        navis_feature = self.norm2(navis_feature)
        
        # 4.6 feedforward network
        navis_feature = self.norm3(self.ffn(navis_feature))
        
        # 4.9 predict the offset & heading
        poses_reg, poses_unc = self.task_decoder(navis_feature) #bs,1,2; bs,1,2
        poses_reg[...,:2] = poses_reg[...,:2] + navis
        
        return poses_reg, poses_unc
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, 
                navis_feature, 
                navis, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                status_encoding,
                ):
        poses_reg_list = []
        poses_unc_list = []
        navis_ = navis
        for mod in self.layers:
            poses_reg, poses_unc = mod(navis_feature, navis_, bev_feature, bev_spatial_shape, agents_query, ego_query, status_encoding)
            poses_reg_list.append(poses_reg)
            poses_unc_list.append(poses_unc)
            navis_ = poses_reg[...,:2].clone().detach()
        return poses_reg_list, poses_unc_list

class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str,config: MimirConfig):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )
        # print(plan_anchor_path)
        # import pdb;pdb.set_trace()
        plan_anchor = np.load(plan_anchor_path)

        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) # 20,8,2
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1,512),
            nn.Linear(d_model, d_model),
        )

        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)

        # self.loss_computer = LossComputer(config)
        self.loss_computer=LaplaceNLLLoss()
        self.training=config.training
        self.use_gt_goal_train=config.use_gt_goal_train
    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2*(odo_info_fut_x + 1.2)/56.9 -1
        odo_info_fut_y = 2*(odo_info_fut_y + 20)/46 -1
        odo_info_fut_head = 2*(odo_info_fut_head + 2)/3.9 -1
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = (odo_info_fut_x + 1)/2 * 56.9 - 1.2
        odo_info_fut_y = (odo_info_fut_y + 1)/2 * 46 - 20
        odo_info_fut_head = (odo_info_fut_head + 1)/2 * 3.9 - 2
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    def forward(self, ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None,goalpoint=None,points_score=1.0) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.training:
            return self.forward_train(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,targets,goalpoint=goalpoint)
        else:
            return self.forward_test(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,global_img,goalpoint=goalpoint,points_score=points_score)


    def forward_train(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding,targets=None,goalpoint=None) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]
        device = ego_query.device
        # 1. add truncated noise to the plan anchor
        navis=goalpoint # (bs,1,2)

        ego_fut_mode = navis.shape[1]
        # 2. proj noisy_traj_points to the query
        navis_pos_embed = gen_sineembed_for_position(navis,hidden_dim=512)
        navis_pos_embed = navis_pos_embed.flatten(-2)
        navis_feature = self.plan_anchor_encoder(navis_pos_embed)
        navis_feature = navis_feature.view(bs,ego_fut_mode,-1) # (bs,1,256)

        # 4. begin the stacked decoder
        poses_reg_list, poses_unc_list = self.diff_decoder(navis_feature, navis, bev_feature, bev_spatial_shape, agents_query, ego_query, status_encoding)

        navi_loss_dict = {}
        unc_navi_loss = 0
        for idx, (poses_reg, poses_unc) in enumerate(zip(poses_reg_list, poses_unc_list)):
            navi_loss = self.loss_computer(poses_reg, poses_unc, targets)
            navi_loss_dict[f"trajectory_loss_{idx}"] = navi_loss
            unc_navi_loss += navi_loss

        best_reg = poses_reg_list[-1].squeeze(1)
        return {"navi": best_reg,"trajectory_loss":unc_navi_loss,"trajectory_loss_dict":navi_loss_dict}

    def forward_test(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding,global_img,goalpoint=None,points_score=1.0) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]
        device = ego_query.device
        # 1. add truncated noise to the plan anchor
        navis=goalpoint

        ego_fut_mode = navis.shape[1]
        # 2. proj noisy_traj_points to the query
        navis_pos_embed = gen_sineembed_for_position(navis,hidden_dim=512)
        navis_pos_embed = navis_pos_embed.flatten(-2)
        navis_feature = self.plan_anchor_encoder(navis_pos_embed)
        navis_feature = navis_feature.view(bs,ego_fut_mode,-1)

        # 4. begin the stacked decoder
        poses_reg_list, poses_unc_list = self.diff_decoder(navis_feature, navis, bev_feature, bev_spatial_shape, agents_query, ego_query, status_encoding)
        best_reg = poses_reg_list[-1].squeeze(1)
        poses_unc = poses_unc_list[-1]
                
        return {"navi": best_reg,
                "unc": poses_unc,
                'anchor_trajectories': goalpoint
                }
    