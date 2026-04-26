export CUDA_VISIBLE_DEVICES="7"
export NAVSIM_DEVKIT_ROOT='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/workspace/Mimir'
export NAVSIM_EXP_ROOT='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/navsim_exp'
export OPENBLAS_CORETYPE=Haswell


# ====================================== use unc to train ==============================================================
TRAIN_TEST_SPLIT=navtest

CHECKPOINT_PATH='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/workspace/socket/navsim_exp/ckpt/mimir_epoch94.ckpt'

GOAL_COORD_PATH='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/workspace/socket/navsim_exp/1_goal_point_unc/navtest_revise/navi.npy'
UNC_PATH='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/workspace/socket/navsim_exp/1_goal_point_unc/navtest_revise/unc.npy'

METRIC_CAHCE_PATH='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/metric_cache_navtest'
TRAJ_SAVE_PATH='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/workspace/socket/navsim_exp/1_goal_point_unc/navtest_plus_revise/trajs'

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
agent=mimir_agent \
train_test_split=$TRAIN_TEST_SPLIT \
experiment_name=a_navtest_mimir_agent_traj_eval \
metric_cache_path=$METRIC_CAHCE_PATH \
worker.threads_per_node=32 \
agent.traj_save_path=$TRAJ_SAVE_PATH \
agent.config.latent=False \
agent.config.training=False \
agent.config.use_proj_image=False \
agent.config.use_gt_goal_train=False \
agent.checkpoint_path=$CHECKPOINT_PATH \
agent.config.status_norm=False \
agent.config.use_unc_score=True \
agent.config.navi_path=$GOAL_COORD_PATH \
agent.config.unc_path=$UNC_PATH