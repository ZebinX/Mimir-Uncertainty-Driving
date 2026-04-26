import numpy as np
import yaml
import copy
import os
from cal_grad import cal_xy_grad
from cal_grad import grid_size,x_range,y_range
from scipy.special import expit
import time
goal_point_coord=np.load('/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/ckpts/cluster_points_8192_.npy')
goal_point_score='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/navsim_exp/1_goalflow_navhard_default/dac'
navhard_yaml='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/workspace/socket/navsim/planning/script/config/common/train_test_split/scene_filter/navhard_two_stage.yaml'
with open(navhard_yaml,'r') as f:
    navhard=yaml.safe_load(f)

def cal_nearest(navi,token):


    dac_scores=np.load(os.path.join(goal_point_score,f'{token}.npy'))
    dac_scores=expit(dac_scores)
    coord_score=np.concatenate([goal_point_coord[...,:2],dac_scores[0]],axis=-1)

    distances = np.sqrt((coord_score[:, 0] - navi[0])**2 + (coord_score[:, 1] - navi[1])**2)
    nearest_index = np.argmin(distances)
    if coord_score[nearest_index][2]>0.5:
        return coord_score[nearest_index][:2]

    valid_points = coord_score[coord_score[:, 2] > 0.5]

    # 2) Calculate Euclidean distance to navi
    distances = np.sqrt((valid_points[:, 0] - navi[0])**2 + (valid_points[:, 1] - navi[1])**2)

    # 3) Find the point with minimum distance (argmin gets index)
    nearest_index = np.argmin(distances)

    return valid_points[nearest_index][:2]


def cal_9th_point(trajectory,time_step):
    # Calculate based on velocity and acceleration of last two points
    x_last, y_last = trajectory[-1]
    x_second_last, y_second_last = trajectory[-2]
    x_third_last, y_third_last = trajectory[-3]

    # Calculate velocity
    vx_last = (x_last - x_second_last) / time_step
    vx_second_last = (x_second_last - x_third_last) / time_step
    vy_last = (y_last - y_second_last) / time_step
    vy_second_last = (y_second_last - y_third_last) / time_step

    # Calculate acceleration
    ax = (vx_last - vx_second_last) / time_step
    ay = (vy_last - vy_second_last) / time_step

    # Predict the 9th point
    x_dir=vx_last * time_step + 0.5 * ax * time_step**2
    y_dir=vy_last * time_step + 0.5 * ay * time_step**2
    x_next = x_last + x_dir
    y_next = y_last + y_dir
    return np.array([x_next,y_next]),np.array([x_dir,y_dir])

def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def matrix_from_pose(pose):
    """
    Converts a 2D pose to a 3x3 transformation matrix

    :param pose: 2D pose (x, y, yaw)
    :return: 3x3 transformation matrix
    """
    return np.array(
        [
            [np.cos(pose[-1]), -np.sin(pose[-1]), pose[0]],
            [np.sin(pose[-1]), np.cos(pose[-1]), pose[1]],
            [0, 0, 1],
        ]
    )

def relative_to_absolute_poses(origin_pose, relative_poses):
    """
    Converts a list of SE2 poses from relative to absolute coordinates using an origin pose.
    :param origin_pose: Reference origin pose
    :param relative_poses: list of relative poses to convert
    :return: list of converted absolute poses
    """
    relative_transforms= np.array([matrix_from_pose(pose) for pose in relative_poses])
    origin_transform = matrix_from_pose(origin_pose)
    absolute_transforms = origin_transform @ relative_transforms
    absolute_poses = np.array([pose_from_matrix(transform_matrix) for transform_matrix in absolute_transforms])

    return absolute_poses

def pose_from_matrix(transform_matrix):

    result=np.zeros((3))
    heading = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
    result[0]=transform_matrix[0,2]
    result[1]=transform_matrix[1,2]
    result[2]=heading

    return result

def convert_absolute_to_relative_point_array(
    origin_array, point_array
):
    """
    Converts an points array from global to relative coordinates.
    :param origin: origin pose of relative coords system
    :param points_array: array of points with (x,y) in last dim
    :return: points coords array in relative coordinates
    """

    theta = -origin_array[-1]

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    points_rel = point_array - origin_array
    points_rel[..., :2] = points_rel[..., :2] @ R.T
    points_rel[..., 2] = normalize_angle(points_rel[:,2])

    return points_rel

def sample_local_points(center, radius, num=25):
    x, y = center
    side = int(np.sqrt(num))
    offsets = np.linspace(-radius, radius, side)
    xs, ys = np.meshgrid(offsets, offsets)
    samples = np.stack([x + xs.ravel(), y + ys.ravel()], axis=-1)
    return samples

def cost_fn(point, pred_point, dac_score, lam=0.3):
    distance = np.linalg.norm(point - pred_point)
    penalty = 1.0 - dac_score
    return distance + lam * penalty

def get_dac_scores_for_points(points, scores_grid_smooth, x_range, y_range):
    """
    Get DAC scores for given points using bilinear interpolation.
    
    Args:
        points: Array of points to query (n, 2)
        scores_grid_smooth: DAC scores grid (grid_size, grid_size)
        x_range: [min_x, max_x] range for the grid
        y_range: [min_y, max_y] range for the grid
        
    Returns:
        scores: DAC scores for each point
    """
    scores = []
    
    # Calculate grid coordinates
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    for point in points:
        x, y = point
        
        # Convert real-world coordinates to grid indices
        i = (x - x_min) / (x_max - x_min) * (grid_size - 1)
        j = (y - y_min) / (y_max - y_min) * (grid_size - 1)

        i = np.clip(i, 0, grid_size - 1)
        j = np.clip(j, 0, grid_size - 1)

        # Get fractional parts
        i_floor = np.floor(i).astype(int)
        i_ceil = np.minimum(i_floor + 1, grid_size - 1)
        j_floor = np.floor(j).astype(int)
        j_ceil = np.minimum(j_floor + 1, grid_size - 1)
        
        # Bilinear interpolation weights
        dx = i - i_floor
        dy = j - j_floor
        
        # Perform bilinear interpolation
        val1 = scores_grid_smooth[j_floor, i_floor] * (1 - dx) * (1 - dy)
        val2 = scores_grid_smooth[j_floor, i_ceil] * dx * (1 - dy)
        val3 = scores_grid_smooth[j_ceil, i_floor] * (1 - dx) * dy
        val4 = scores_grid_smooth[j_ceil, i_ceil] * dx * dy
        
        scores.append(val1 + val2 + val3 + val4)
    
    return np.array(scores)

frame_mapping_path='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/data/frame_mapping.yaml'
navi_path='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/navsim_exp/1_goal_point_unc/navhard_default/navi.npy'
global_navi_path='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/navsim_exp/1_goal_point_unc/global_pose.npy'
traj_dir='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/workspace/socket/navsim_exp/1_traj_test'

# token='0a8cbe192f3d3dfba'

global_navi=np.load(global_navi_path,allow_pickle=True).item()
navi=np.load(navi_path,allow_pickle=True).item()
result=copy.deepcopy(navi)
with open(frame_mapping_path,'r') as f:
    frame_mapping=yaml.safe_load(f)

start_time=time.time()
for key in frame_mapping.keys():
    traj=np.load(os.path.join(traj_dir,f'{key}.npy'))
    navi,inter_dir=cal_9th_point(traj[...,:2],time_step=0.5)

    # if key in navhard['tokens']:
    #     navi=cal_nearest(navi,key)
    cal_nearest(navi,key)

    origin_relative_pose=np.zeros((1,3))
    origin_relative_pose[0,:2]=navi
    global_pose1=global_navi[key]
    absolute_pose=relative_to_absolute_poses(global_pose1,origin_relative_pose)
    global_pose2=global_navi[frame_mapping[key]]
    convert_relative_pose=convert_absolute_to_relative_point_array(global_pose2,absolute_pose)
    result[frame_mapping[key]]=convert_relative_pose[0,:2]
end_time=time.time()
print((end_time-start_time)*1000)


# np.save('/lpai/socket/navsim_exp/1_goal_point_unc/navhard_default/navhard_nearest_recursion_real.npy',result)