from typing import List, Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)

def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers

def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
    """
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos

def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


class GridSampleCrossBEVAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, num_levels=1, in_bev_dims=64, num_points=8, config=None):
        super(GridSampleCrossBEVAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.config = config
        self.attention_weights = nn.Linear(embed_dims,num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)


        self.value_proj = nn.Sequential(
            nn.Conv2d(in_bev_dims, 256, kernel_size=(3, 3), stride=(1, 1), padding=1,bias=True),
            nn.ReLU(inplace=True),
        )

        self.init_weight()

    def init_weight(self):

        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)


    def forward(self, queries, traj_points, bev_feature, spatial_shape):
        """
        Args:
            queries: input features with shape of (bs, num_queries, embed_dims)
            traj_points: trajectory points with shape of (bs, num_queries, num_points, 2)
            bev_feature: bev features with shape of (bs, embed_dims, height, width)
            spatial_shapes: (height, width)

        """

        bs, num_queries, num_points, _ = traj_points.shape
        
        # Normalize trajectory points to [-1, 1] range for grid_sample
        normalized_trajectory = traj_points.clone()
        normalized_trajectory[..., 0] = normalized_trajectory[..., 0] / self.config.lidar_max_y
        normalized_trajectory[..., 1] = normalized_trajectory[..., 1] / self.config.lidar_max_x

        normalized_trajectory = normalized_trajectory[..., [1, 0]]  # Swap x and y
        
        attention_weights = self.attention_weights(queries)
        attention_weights = attention_weights.view(bs, num_queries, num_points).softmax(-1)

        value = self.value_proj(bev_feature)
        grid = normalized_trajectory.view(bs, num_queries, num_points, 2)
        # Sample features
        sampled_features = torch.nn.functional.grid_sample(
            value, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        ) # bs, C, num_queries, num_points

        attention_weights = attention_weights.unsqueeze(1)
        out = (attention_weights * sampled_features).sum(dim=-1)
        out = out.permute(0, 2, 1).contiguous()  # bs, num_queries, C
        out = self.output_proj(out)

        return self.dropout(out) + queries


# navi
class GridSampleCrossBEVAttention_navi(nn.Module):
    def __init__(self, embed_dims, num_heads, num_levels=1, in_bev_dims=64, num_points=1, config=None):
        super(GridSampleCrossBEVAttention_navi, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.config = config
        self.attention_weights = nn.Linear(embed_dims,num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)


        self.value_proj = nn.Sequential(
            nn.Conv2d(in_bev_dims, 256, kernel_size=(3, 3), stride=(1, 1), padding=1,bias=True),
            nn.ReLU(inplace=True),
        )

        self.init_weight()

    def init_weight(self):

        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)


    def forward(self, queries, navi_points, bev_feature, spatial_shape, point_score=1.0):
        """
        Args:
            queries: input features with shape of (bs, num_queries, embed_dims)
            traj_points: trajectory points with shape of (bs, num_queries, num_points, 2)
            bev_feature: bev features with shape of (bs, embed_dims, height, width)
            spatial_shapes: (height, width)

        """
        # print(f"gt_point:{gt_points.shape}")
        # print(f"queries:{queries.shape}")
        # import pdb;pdb.set_trace()
        batch_size=bev_feature.shape[0]
        if not torch.is_tensor(navi_points):
            navi_points = torch.from_numpy(navi_points).float()
            navi_points=navi_points.to(torch.float32)
            navi_points=navi_points.unsqueeze(0)
            navi_points=navi_points.unsqueeze(1)
            navi_points=navi_points.unsqueeze(1)
            navi_points=navi_points.expand(-1,1280,-1,-1)
        else:
            navi_points=navi_points.view(batch_size,1,1,2)
            navi_points=navi_points.expand(-1,20,-1,-1)
        # 1 1280 1 2
        bs, num_queries, num_points, _ = navi_points.shape
        
        # Normalize trajectory points to [-1, 1] range for grid_sample
        normalized_trajectory = navi_points.clone()
        normalized_trajectory[..., 0] = normalized_trajectory[..., 0] / self.config.lidar_max_y
        normalized_trajectory[..., 1] = normalized_trajectory[..., 1] / self.config.lidar_max_x

        normalized_trajectory = normalized_trajectory[..., [1, 0]]  # Swap x and y
        
        # 64 20 1
        attention_weights = self.attention_weights(queries)
        attention_weights = attention_weights.view(bs, num_queries, num_points).softmax(-1)

        value = self.value_proj(bev_feature) # bs 256 64 64

        #64 20 1 2
        grid = normalized_trajectory.view(bs, num_queries, num_points, 2)
        grid=grid.to(torch.float32)
        # Sample features
        sampled_features = torch.nn.functional.grid_sample(
            value, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        ) # bs, C, num_queries, num_points

        # print("point_score==============================================",point_score)
        # 64 1 1 1 
        attention_weights = attention_weights.unsqueeze(1)*point_score # (bs,1,num_queries,1)
        out = (attention_weights * sampled_features).sum(dim=-1)
        out = out.permute(0, 2, 1).contiguous()  # bs, num_queries, C
        out = self.output_proj(out)

        return self.dropout(out) + queries

class GridSampleCrossBEVAttention_naviscore(nn.Module):
    def __init__(self, embed_dims, num_heads, num_levels=1, in_bev_dims=64, num_points=1, config=None):
        super(GridSampleCrossBEVAttention_naviscore, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.config = config
        self.attention_weights = nn.Linear(embed_dims,num_points)
        self.attention_weights_score = nn.Linear(embed_dims,num_points)
        # self.score_encoding=gen_sineembed_for_position()
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)


        self.value_proj = nn.Sequential(
            nn.Conv2d(in_bev_dims, 256, kernel_size=(3, 3), stride=(1, 1), padding=1,bias=True),
            nn.ReLU(inplace=True),
        )

        self.init_weight()
        self.point_encoding=SinusoidalPosEmb(256)

    def init_weight(self):

        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)


    def forward(self, queries, navi_points, bev_feature, spatial_shape, point_score=1.0):
        """
        Args:
            queries: input features with shape of (bs, num_queries, embed_dims)
            traj_points: trajectory points with shape of (bs, num_queries, num_points, 2)
            bev_feature: bev features with shape of (bs, embed_dims, height, width)
            spatial_shapes: (height, width)

        """
        # print(f"gt_point:{gt_points.shape}")
        # print(f"queries:{queries.shape}")
        # import pdb;pdb.set_trace()
        batch_size=bev_feature.shape[0]
        if not self.training:
            navi_points=navi_points.view(batch_size,1,1,2)
            navi_points=navi_points.expand(-1,1280,-1,-1)
        else:
            navi_points=navi_points.view(batch_size,1,1,2)
            navi_points=navi_points.expand(-1,20,-1,-1)
        # 1 1280 1 2
        bs, num_queries, num_points, _ = navi_points.shape
        
        # Normalize trajectory points to [-1, 1] range for grid_sample
        normalized_trajectory = navi_points.clone()
        normalized_trajectory[..., 0] = normalized_trajectory[..., 0] / self.config.lidar_max_y
        normalized_trajectory[..., 1] = normalized_trajectory[..., 1] / self.config.lidar_max_x

        normalized_trajectory = normalized_trajectory[..., [1, 0]]  # Swap x and y
        
        # 64 20 1
        attention_weights = self.attention_weights(queries)
        attention_weights = attention_weights.view(bs, num_queries, num_points).softmax(-1)

        point_score=point_score.view(bs,1,1,2)
        point_score=point_score.expand(-1,num_queries,-1,-1)
        attention_weights_score = self.attention_weights_score(gen_sineembed_for_position(point_score))
        attention_weights_score = attention_weights_score.view(bs,num_queries,num_points)
        

        value = self.value_proj(bev_feature) # bs 256 64 64

        #64 20 1 2
        grid = normalized_trajectory.view(bs, num_queries, num_points, 2)
        grid=grid.to(torch.float32)
        # Sample features
        sampled_features = torch.nn.functional.grid_sample(
            value, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        ) # bs, C, num_queries, num_points

        # print("point_score==============================================",point_score)
        # 64 1 1 1 
        # attention_weights = attention_weights.unsqueeze(1)*point_score # (bs,1,num_queries,1)
        attention_weights=attention_weights.unsqueeze(1)
        attention_weights_score=attention_weights_score.unsqueeze(1)
        out = (attention_weights * sampled_features * attention_weights_score).sum(dim=-1)
        out = out.permute(0, 2, 1).contiguous()  # bs, num_queries, C
        out = self.output_proj(out)

        return self.dropout(out) + queries

# navi unc
class GridSampleCrossBEVAttention_naviunc(nn.Module):
    def __init__(self, embed_dims, num_heads, num_levels=1, in_bev_dims=64, num_points=1, config=None):
        super(GridSampleCrossBEVAttention_naviunc, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.config = config
        self.attention_weights = nn.Linear(embed_dims,num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(0.1)


        self.value_proj = nn.Sequential(
            nn.Conv2d(in_bev_dims, 256, kernel_size=(3, 3), stride=(1, 1), padding=1,bias=True),
            nn.ReLU(inplace=True),
        )

        self.init_weight()

    def init_weight(self):

        nn.init.constant_(self.attention_weights.weight, 0)
        nn.init.constant_(self.attention_weights.bias, 0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0)


    def forward(self, queries, navi_points, bev_feature, spatial_shape, point_score=1.0):
        """
        Args:
            queries: input features with shape of (bs, num_queries, embed_dims)
            traj_points: trajectory points with shape of (bs, num_queries, num_points, 2)
            bev_feature: bev features with shape of (bs, embed_dims, height, width)
            spatial_shapes: (height, width)

        """

        batch_size=bev_feature.shape[0]

        navi_points=navi_points
        bs, num_points, _ = navi_points.shape
        
        # Normalize trajectory points to [-1, 1] range for grid_sample
        normalized_trajectory = navi_points.clone()
        normalized_trajectory[..., 0] = normalized_trajectory[..., 0] / self.config.lidar_max_y
        normalized_trajectory[..., 1] = normalized_trajectory[..., 1] / self.config.lidar_max_x

        normalized_trajectory = normalized_trajectory[..., [1, 0]]  # Swap x and y
        
        attention_weights = self.attention_weights(queries)
        attention_weights = attention_weights.view(bs, 1, num_points).softmax(-1)

        value = self.value_proj(bev_feature) # bs 256 64 64

        #64 20 1 2
        grid = normalized_trajectory.view(bs, num_points,1, 2)
        grid=grid.to(torch.float32)
        # Sample features
        sampled_features = torch.nn.functional.grid_sample(
            value, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        ) # bs, C, num_queries, num_points

        attention_weights = attention_weights.unsqueeze(1)*point_score
        out = (attention_weights * sampled_features).sum(dim=-1)
        out = out.permute(0, 2, 1).contiguous()  # bs, num_queries, C
        out = self.output_proj(out)

        return self.dropout(out) + queries
