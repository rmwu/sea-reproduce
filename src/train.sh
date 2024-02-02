# main script for training

######### training params
CUDA=0
NUM_GPU=1

######### data params

# NOTE: name of YAML file and run save folder
# see ./config for more options
TAG="aggregator_tf_fci"
#TAG="aggregator_tf_gies"
CONFIG="config/${TAG}.yaml"

# NOTE: customize this to your save folder
# it'll make a new subfolder within with timestamp
SAVE_PATH="/data/scratch/rmwu/tmp-runs/sc-baselines/${TAG}"
# if you messed up and your job died, uncomment this and --checkpoint_path
#CKPT_PATH=""

python src/train.py \
    --config_file $CONFIG \
    --save_path $SAVE_PATH \
    --gpu $CUDA \
    --num_gpu $NUM_GPU \
    #--checkpoint_path $CKPT_PATH \

