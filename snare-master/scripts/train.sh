EXP_NAME=$1
CFG=$2
LAGOR_CKPT=$4

WANDB_MODE=offline python train.py --config-name $CFG train.rotator.pretrained_cls=$LAGOR_CKPT debug=False train.random_seed=$RANDOM_SEED train.exp_name=$EXP_NAME train.exps_folder=snap/ train.log=True
