export OPENBLAS_CORETYPE=Haswell
export NAVSIM_DEVKIT_ROOT='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/workspace/Mimir'
export NAVSIM_EXP_ROOT='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/navsim_exp'
CACHE_TO_SAVE='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/mimir_agent_feature_cache' #set your feature cache path to save
TRAIN_TEST_SPLIT=navtrain
export HYDRA_FULL_ERROR=1

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
agent=mimir_agent \
experiment_name=a_mimir_trainval_feature_cache \
cache_path=$CACHE_TO_SAVE \
train_test_split=$TRAIN_TEST_SPLIT \
agent.config.latent=False \
worker.threads_per_node=32