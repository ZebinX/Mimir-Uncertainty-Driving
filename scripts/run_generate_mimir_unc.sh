# ======================================================= navtest ==============================================================================
TRAIN_TEST_SPLIT=navtest
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=4

export NAVSIM_DEVKIT_ROOT='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/workspace/Mimir'
export NAVSIM_EXP_ROOT='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/navsim_exp_mimir'
export OPENBLAS_CORETYPE=Haswell

CHECKPOINT_PATH='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/workspace/socket/navsim_exp/ckpt/mimir_unc_epoch99.ckpt'
CACHE_PATH=/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/metric_cache_navtest
GOAL_COORD_PATH='/lpai/dataset/navsim-exp-mimir/25-12-23-1/navsim_exp/1_goal_point_coords/navtest_default.npy'


python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_generate_unc_navtest.py \
agent=mimir_agent_unc \
experiment_name=a_navtest_mimir_agent_unc_eval \
train_test_split=$TRAIN_TEST_SPLIT \
agent.config.latent=False \
agent.config.training=False \
agent.config.use_proj_image=False \
agent.config.use_gt_goal_train=False \
agent.checkpoint_path=$CHECKPOINT_PATH \
agent.config.status_norm=False \
agent.config.navi_bank_path='/lpai/dataset/navsim-exp-mimir/25-12-23-1/navsim_exp/1_goal_point_coords/navtest_default.npy' \
agent.config.navi_unc_outputdir='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/workspace/socket/navsim_exp/tmp' \
agent.config.goal_coord_path=$GOAL_COORD_PATH \
metric_cache_path=$CACHE_PATH \
worker.threads_per_node=32