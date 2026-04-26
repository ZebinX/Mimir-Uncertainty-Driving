import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
import os
from scipy.special import expit

# Step 1: Construct grid (assuming coordinate system range in [0, 10])
grid_size = 256  # Grid resolution
x_range = [0, 65]
y_range = [-30,30]
x = np.linspace(x_range[0], x_range[1], grid_size)
y = np.linspace(y_range[0], y_range[1], grid_size)
xv, yv = np.meshgrid(x, y)  # 创建网格

# Step 2: Map goal points to the grid
# goal_points = np.random.rand(8192, 3)  # Assume randomly generated goal points, (x, y, score)
goal_point_coord=np.load('/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/ckpts/cluster_points_8192_.npy')

def cal_xy_grad(token,goal_point_score_dir='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/navsim_exp/1_goalflow_navhard_default/dac'):
    goal_point_score=np.load(os.path.join(goal_point_score_dir,f'{token}.npy'))
    goal_points=np.concatenate([goal_point_coord[...,:2],goal_point_score[0]],axis=-1)
    # goal_points[:, :2]

    # Create score distribution on grid
    scores_grid = np.zeros((grid_size, grid_size))
    num_grid = np.zeros((grid_size, grid_size))+1e-6
    for point in goal_points:
        x_idx = int((point[0] - x_range[0]) / (x_range[1] - x_range[0]) * grid_size)
        y_idx = int((point[1] - y_range[0]) / (y_range[1] - y_range[0]) * grid_size)
        scores_grid[y_idx, x_idx] += point[2]  # Add score to nearest grid point
        num_grid[y_idx,x_idx]+=1
    scores_grid=scores_grid/num_grid

    # Step 3: Smooth scores (Gaussian filter)
    scores_grid_smooth = gaussian_filter(scores_grid, sigma=1)  # Smooth score distribution
    # scores_grid_smooth=expit(scores_grid_smooth)

    # Construct coordinates (corresponding to grid)
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

    # Step 4: Calculate gradient field
    gradient_x = sobel(scores_grid_smooth, axis=0)  # Gradient in x-axis direction
    gradient_y = sobel(scores_grid_smooth, axis=1)  # Gradient in y-axis direction

    return gradient_x,gradient_y,scores_grid_smooth

    # gy, gx = np.gradient(scores_grid_smooth, edge_order=2)
    # # 3）梯度幅值与方向
    # magnitude = np.hypot(gx, gy)
    # angle     = np.arctan2(gy, gx)
    # return gy,gx,magnitude,angle


token='5c4d8dfcb0aa5541'
cal_xy_grad(token)

# # Visualize gradient field
# plt.imshow(scores_grid_smooth, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), origin='lower')
# plt.quiver(xv, yv, gradient_y, -gradient_x, color='red')  # Gradient field arrows
# plt.title("Gradient Field with Goal Points")
# plt.savefig(f'/lpai/socket/tools/cal_grad/figs/{token}_grad.png')


# version 2
# Demo visualization
# plt.figure(figsize=(40,12))
# plt.subplot(131); plt.title('gx'); plt.imshow(gx, cmap='RdBu'); plt.colorbar()
# plt.subplot(132); plt.title('gy'); plt.imshow(gy, cmap='RdBu'); plt.colorbar()
# plt.subplot(133); plt.title('magnitude'); plt.imshow(gradient_x, cmap='gray'); plt.colorbar()
# plt.tight_layout()
# plt.savefig(f'/lpai/socket/tools/cal_grad/figs/{token}_grad.png',dpi=300)