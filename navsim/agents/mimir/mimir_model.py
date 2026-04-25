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
from navsim.agents.mimir.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention,GridSampleCrossBEVAttention_navi,GridSampleCrossBEVAttention_naviscore
from navsim.agents.mimir.modules.multimodal_loss import LossComputer
from torch.nn import TransformerDecoder,TransformerDecoderLayer
from typing import Any, List, Dict, Optional, Union,Tuple
import numpy.typing as npt


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

def load_navis_from_np(data_dict,token):
    if isinstance(token,str):
        data=[]
        data.append(data_dict[token])
        data=np.stack(data,axis=0)
        data=torch.from_numpy(data).unsqueeze(1)
    else:
        data=[]
        for t in token:
            data.append(data_dict[t])
        data=np.stack(data,axis=0)
        data=torch.from_numpy(data).unsqueeze(1)
    return data


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
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1,320),
        )
        self.weight_score=None
        if self._config.unc_path:
            self.weight_score=np.load(self._config.unc_path,allow_pickle=True).item()

        self.goalpoints=None
        if self._config.navi_path:
            self.goalpoints=np.load(self._config.navi_path,allow_pickle=True).item()
    
    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None,goalpoint=None,token=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        
        if self._config.latent:
            lidar_feature = None
        else:
            lidar_feature: torch.Tensor = features["lidar_feature"]
        # lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]
        if self._config.status_norm and self._config.training==False:
            vle = torch.norm(status_feature[:,4:6],dim=-1,keepdim=True)
            acc = torch.norm(status_feature[:,6:8],dim=-1,keepdim=True)
            dot_product = torch.sum(status_feature[:,4:6]*status_feature[:,6:8],dim=-1,keepdim=True)
            tag_vle = torch.where(status_feature[:,4:5] >= 0,1.0,-1.0)
            tag_acc = torch.where(dot_product > 0,1.0,-1.0)
            status_feature= torch.cat([status_feature[:,:4], tag_vle*vle,tag_acc*acc], dim=-1)
        
        # 只用command
        # status_feature=status_feature[:,:4]
        
        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, img_feature = self._backbone(camera_feature, lidar_feature)
        if self._config.training and self._config.use_proj_image:
            navis=features['gt_trajs'][:,-1:,:]
            navis[:,:,2]=0.0
            R=features['sensor2lidar_rot'][:,0].to(navis)
            T=features['sensor2lidar_trans'][:,0].to(navis)
            K=features['intrinsic'][:,0].to(navis)
            pix=extract_feature_values_at_navi_batched(img_feature[:,:,:,17:-17],navis,R,T,K,(1024,1920))
        elif self._config.use_proj_image:
            R=features['sensor2lidar_rot'][:,0].to(camera_feature)
            T=features['sensor2lidar_trans'][:,0].to(camera_feature)
            K=features['intrinsic'][:,0].to(camera_feature)
            goalpoint_tensor=torch.from_numpy(goalpoint).to(camera_feature).unsqueeze(0).unsqueeze(0)
            zeros=torch.zeros_like(goalpoint_tensor)
            goalpoint_tensor=torch.cat([goalpoint_tensor,zeros[:,:,-1:]],dim=-1)
            pix=extract_feature_values_at_navi_batched(img_feature[:,:,:,17:-17],goalpoint_tensor,R,T,K,(1024,1920))
        else:
            pix=None
        # pix.shape [bs,256,1]

        if self.weight_score:
            if self._config.training:
                points_score=load_navis_from_np(self.weight_score,features['token']).to(bev_feature)
            else:
                points_score=load_navis_from_np(self.weight_score,token).to(bev_feature)
            # points_score=0.0
        else:
            points_score=None
        
        if self.goalpoints:
            if self._config.training:
                goalpoint=load_navis_from_np(self.goalpoints,features['token']).to(bev_feature)
            else:
                goalpoint=load_navis_from_np(self.goalpoints,token).to(bev_feature)
        else:
            goalpoint=None
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

        trajectory = self._trajectory_head(trajectory_query,agents_query, cross_bev_feature,bev_spatial_shape,status_encoding[:, None],targets=targets,global_img=pix,goalpoint=goalpoint,points_score=points_score)
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

class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
    def forward(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape

        # 6. get final prediction
        traj_feature = traj_feature.view(bs, ego_fut_mode,-1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs,ego_fut_mode, self.ego_fut_ts, 3)

        return plan_reg, plan_cls
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
        if config.use_unc_score==True:
            self.cross_bev_attention_navi= GridSampleCrossBEVAttention_naviscore(
                config.tf_d_model,
                config.tf_num_head,
                num_points=1,
                config=config,
                in_bev_dims=256,
            )
        else:
            self.cross_bev_attention_navi = GridSampleCrossBEVAttention_navi(
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
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=20,
        )

    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None,
                navi_points=None,
                points_score=1.0,
                ):
        if self.use_proj_image:
            global_img=global_img.permute(0,2,1)
            traj_feature = traj_feature+self.dropout2(self.cross_imgnavi_attention(traj_feature,global_img,global_img)[0])
            traj_feature=self.norm4(traj_feature)
            
        traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
        if navi_points is not None:
            traj_feature = self.cross_bev_attention_navi(traj_feature,navi_points,bev_feature,bev_spatial_shape,points_score)
        
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query,agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        
        # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])

        # 4.5 cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query,ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        
        # 4.6 feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        # 4.8 modulate with time steps
        traj_feature = self.time_modulation(traj_feature, time_embed,global_cond=None,global_img=None)
        
        # 4.9 predict the offset & heading
        poses_reg, poses_cls = self.task_decoder(traj_feature) #bs,20,8,3; bs,20
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi
        
        return poses_reg, poses_cls
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
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None,
                navi_points=None,
                points_score=1.0,
                ):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img,navi_points,points_score=points_score)
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[...,:2].clone().detach()
        return poses_reg_list, poses_cls_list

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
        self.goalpoint_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1,256),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)

        self.loss_computer = LossComputer(config)
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
            return self.forward_train(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,targets,global_img,goalpoint=goalpoint,points_score=points_score)
        else:
            return self.forward_test(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,global_img,goalpoint=goalpoint,points_score=points_score)


    def forward_train(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None,goalpoint=None,points_score=None) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]
        device = ego_query.device
        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
        odo_info_fut = self.norm_odo(plan_anchor)
        timesteps = torch.randint(
            0, 50,
            (bs,), device=device
        )
        noise = torch.randn(odo_info_fut.shape, device=device)
        noisy_traj_points = self.diffusion_scheduler.add_noise(
            original_samples=odo_info_fut,
            noise=noise,
            timesteps=timesteps,
        ).float()
        noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        noisy_traj_points = self.denorm_odo(noisy_traj_points)

        ego_fut_mode = noisy_traj_points.shape[1]
        # 2. proj noisy_traj_points to the query
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        goal_info_embed=gen_sineembed_for_position(torch.cat([goalpoint,points_score],dim=-2),hidden_dim=128)
        goal_info_embed=goal_info_embed.flatten(-2)
        goal_info_feature=self.goalpoint_encoder(goal_info_embed)
        traj_feature = traj_feature.view(bs,ego_fut_mode,-1)+goal_info_feature.view(bs,1,-1)
        # traj_feature = traj_feature.view(bs,ego_fut_mode,-1)
        # 3. embed the timesteps
        time_embed = self.time_mlp(timesteps)
        time_embed = time_embed.view(bs,1,-1)

        # if self.use_gt_goal_train:
        #     navi_points = targets['trajectory'][:,-1,:2]
        # else:
        #     navi_points = targets['navi_points']
        #     # navi_points= torch.zeros_like(targets['trajectory'][:,-1,:2])
        # alpha=0.3
        # gt_points=targets['trajectory'][:,-1,:2]
        # distances=torch.norm(gt_points-navi_points,dim=1).to(traj_feature)
        # points_score=torch.exp(-alpha*distances)
        navi_points=goalpoint
        # 4. begin the stacked decoder
        poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img,navi_points,points_score)

        trajectory_loss_dict = {}
        ret_traj_loss = 0
        for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
            trajectory_loss = self.loss_computer(poses_reg, poses_cls, targets, plan_anchor)
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = trajectory_loss
            ret_traj_loss += trajectory_loss

        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)
        return {"trajectory": best_reg,"trajectory_loss":ret_traj_loss,"trajectory_loss_dict":trajectory_loss_dict}

    def forward_test(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding,global_img,goalpoint=None,points_score=1.0) -> Dict[str, torch.Tensor]:
        step_num = 2
        bs = ego_query.shape[0]
        anchor = 20
        num_samples = 64
        device = ego_query.device
        self.diffusion_scheduler.set_timesteps(1000, device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)


        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
        plan_anchor = plan_anchor.unsqueeze(2).repeat(1,1,num_samples,1,1)
        plan_anchor = plan_anchor.view(bs,num_samples*anchor, 8, 2)
        img = self.norm_odo(plan_anchor)
        noise = torch.randn(img.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
        noisy_trajs = self.denorm_odo(img)
        ego_fut_mode = img.shape[1]
        for k in roll_timesteps[:]:
            x_boxes = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            goal_info_embed=gen_sineembed_for_position(torch.cat([goalpoint,points_score],dim=-2),hidden_dim=128)
            goal_info_embed=goal_info_embed.flatten(-2)
            goal_info_feature=self.goalpoint_encoder(goal_info_embed)
            traj_feature = traj_feature.view(bs,ego_fut_mode,-1)+goal_info_feature.view(bs,1,-1)

            timesteps = k
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(img.device)
            
            # 3. embed the timesteps
            timesteps = timesteps.expand(img.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs,1,-1)

            # 4. begin the stacked decoder
            navi_points = goalpoint
            poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img,navi_points,points_score)
            poses_reg = poses_reg_list[-1] 
            poses_cls = poses_cls_list[-1]
            x_start = poses_reg[...,:2]
            x_start = self.norm_odo(x_start)
            img = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=img
            ).prev_sample
        # poses_cls = poses_cls.view(bs, anchor*num_samples)  
        mode_idx = poses_cls.argmax(dim=-1)  
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)

        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
        poses_reg = poses_reg.view(bs, anchor, num_samples, self._num_poses, 3)
                
        return {"trajectory": best_reg,
                'anchor_trajectories': poses_reg
                }
    