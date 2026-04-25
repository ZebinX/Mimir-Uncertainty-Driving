TRAIN_TEST_SPLIT=navtrain
export CUDA_VISIBLE_DEVICES="3"

export HYDRA_FULL_ERROR=1
export NAVSIM_DEVKIT_ROOT='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/workspace/Mimir'
export NAVSIM_EXP_ROOT='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/navsim_exp'
export OPENBLAS_CORETYPE=Haswell
CACHE_PATH='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/mimir_agent_feature_cache'
COORD_PATH='/lpai/dataset/navsim-exp-mimir/25-12-23-1/navsim_exp/1_goal_point_coords/navtrain_default.npy'


python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=mimir_agent_unc \
experiment_name=a_navtrain_mimir_agent_unc \
train_test_split=$TRAIN_TEST_SPLIT \
use_cache_without_dataset=True \
cache_path=$CACHE_PATH \
dataloader.params.batch_size=64 \
trainer.params.max_epochs=100 \
force_cache_computation=False \
agent.config.navi_bank_path=$COORD_PATH \
agent.config.latent=False \
agent.config.training=True \
agent.config.use_proj_image=False \
agent.config.use_gt_goal_train=False \
agent.config.status_norm=False