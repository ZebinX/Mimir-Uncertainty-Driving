import numpy as np
import yaml
import copy
import os
import time

def cal_9th_point(trajectory,time_step):
    # 基于最后两点的速度和加速度进行推算
    x_last, y_last = trajectory[-1]
    x_second_last, y_second_last = trajectory[-2]
    x_third_last, y_third_last = trajectory[-3]

    # 计算速度
    vx_last = (x_last - x_second_last) / time_step
    vx_second_last = (x_second_last - x_third_last) / time_step
    vy_last = (y_last - y_second_last) / time_step
    vy_second_last = (y_second_last - y_third_last) / time_step

    # 计算加速度
    ax = (vx_last - vx_second_last) / time_step
    ay = (vy_last - vy_second_last) / time_step

    # 推算第 9 个点
    x_next = x_last + (vx_last * time_step + 0.5 * ax * time_step**2)*1.15
    y_next = y_last + (vy_last * time_step + 0.5 * ay * time_step**2)*1.15
    return np.array([x_next,y_next])

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
    navi=cal_9th_point(traj[...,:2],time_step=0.5)

    origin_relative_pose=np.zeros((1,3))
    origin_relative_pose[0,:2]=navi
    global_pose1=global_navi[key]
    absolute_pose=relative_to_absolute_poses(global_pose1,origin_relative_pose)
    global_pose2=global_navi[frame_mapping[key]]
    convert_relative_pose=convert_absolute_to_relative_point_array(global_pose2,absolute_pose)
    result[frame_mapping[key]]=convert_relative_pose[0,:2]
end_time=time.time()
print((end_time-start_time)*1000)

# np.save('/lpai/socket/navsim_exp/1_goal_point_unc/navhard_default/navhard_phy_recursion1.15.npy',result)