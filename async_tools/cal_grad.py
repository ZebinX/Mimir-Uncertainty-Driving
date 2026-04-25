import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
import os
from scipy.special import expit

# Step 1: 构造网格 (假设坐标系范围在 [0, 10])
grid_size = 256  # 网格分辨率
x_range = [0, 65]
y_range = [-30,30]
x = np.linspace(x_range[0], x_range[1], grid_size)
y = np.linspace(y_range[0], y_range[1], grid_size)
xv, yv = np.meshgrid(x, y)  # 创建网格

# Step 2: 将 goal points 映射到网格上
# goal_points = np.random.rand(8192, 3)  # 假设随机生成 goal points，(x, y, score)
goal_point_coord=np.load('/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/ckpts/cluster_points_8192_.npy')

def cal_xy_grad(token,goal_point_score_dir='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/navsim_exp/1_goalflow_navhard_default/dac'):
    goal_point_score=np.load(os.path.join(goal_point_score_dir,f'{token}.npy'))
    goal_points=np.concatenate([goal_point_coord[...,:2],goal_point_score[0]],axis=-1)
    # goal_points[:, :2]

    # 创建网格上的分数分布
    scores_grid = np.zeros((grid_size, grid_size))
    num_grid = np.zeros((grid_size, grid_size))+1e-6
    for point in goal_points:
        x_idx = int((point[0] - x_range[0]) / (x_range[1] - x_range[0]) * grid_size)
        y_idx = int((point[1] - y_range[0]) / (y_range[1] - y_range[0]) * grid_size)
        scores_grid[y_idx, x_idx] += point[2]  # 将分数加到最近的网格点上
        num_grid[y_idx,x_idx]+=1
    scores_grid=scores_grid/num_grid

    # Step 3: 对分数进行平滑 (高斯滤波)
    scores_grid_smooth = gaussian_filter(scores_grid, sigma=1)  # 平滑化分数分布
    # scores_grid_smooth=expit(scores_grid_smooth)

    # 构造横纵坐标（和网格对应）
    # xs = np.linspace(x_range[0], x_range[1], grid_size)
    # ys = np.linspace(y_range[0], y_range[1], grid_size)
    # X, Y = np.meshgrid(xs, ys)
    # plt.figure(figsize=(6,5))
    # origin='lower'
    # plt.imshow(scores_grid_smooth,
    #         origin='lower',
    #         cmap='viridis',
    #         extent=(x_range[0], x_range[1],
    #                 y_range[0], y_range[1]),
    #         aspect='equal')
    # plt.colorbar(label='smoothed score')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title(f'scores_grid_smooth (token={token})')
    # plt.tight_layout()
    # plt.savefig(f'/lpai/socket/tools/cal_grad/figs/{token}_score.png')

    # Step 4: 计算梯度场
    gradient_x = sobel(scores_grid_smooth, axis=0)  # x 轴方向梯度
    gradient_y = sobel(scores_grid_smooth, axis=1)  # y 轴方向梯度

    return gradient_x,gradient_y,scores_grid_smooth

    # gy, gx = np.gradient(scores_grid_smooth, edge_order=2)
    # # 3）梯度幅值与方向
    # magnitude = np.hypot(gx, gy)
    # angle     = np.arctan2(gy, gx)
    # return gy,gx,magnitude,angle


token='5c4d8dfcb0aa5541'
cal_xy_grad(token)

# # 可视化梯度场
# plt.imshow(scores_grid_smooth, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), origin='lower')
# plt.quiver(xv, yv, gradient_y, -gradient_x, color='red')  # 梯度场箭头
# plt.title("Gradient Field with Goal Points")
# plt.savefig(f'/lpai/socket/tools/cal_grad/figs/{token}_grad.png')


# version 2
# 演示可视化
# plt.figure(figsize=(40,12))
# plt.subplot(131); plt.title('gx'); plt.imshow(gx, cmap='RdBu'); plt.colorbar()
# plt.subplot(132); plt.title('gy'); plt.imshow(gy, cmap='RdBu'); plt.colorbar()
# plt.subplot(133); plt.title('magnitude'); plt.imshow(gradient_x, cmap='gray'); plt.colorbar()
# plt.tight_layout()
# plt.savefig(f'/lpai/socket/tools/cal_grad/figs/{token}_grad.png',dpi=300)