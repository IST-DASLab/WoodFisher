DATASET=imagenet
MODEL=mobilenet
DATA_PATH=../datasets/ILSVRC
CONFIG_PATH=./configs/mobilenetv1.yaml
PRUNER=globalmagni
FITTABLE=10000
TARGET=0.89
EPOCHS=100
FISHER_SUBSAMPLE=400
FISHER_MINI=2400
MAX_MINI_BSZ=800
EXP_NAME="exp_str_prune_($MODEL)_gradmask_($TARGET)_epochs_($EPOCHS)_($PRUNER)_fit_($FITTABLE)_fisher_($FISHER_SUBSAMPLE)_($FISHER_MINI)_final_all_layers"
RESULT_PATH="./logs/$EXP_NAME.log"
RESULT_CSV="./csv/$EXP_NAME.csv"
LOAD_FROM="./checkpoints/MobileNetV1-Dense-STR.pth"

echo "EXPERIMENT $EXP_NAME"

export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=5,4,6,7 python main.py \
--exp_name=$EXP_NAME \
--dset=$DATASET \
--dset_path=$DATA_PATH \
--arch=$MODEL \
--config_path=$CONFIG_PATH \
--workers=24 --batch_size=180 --logging_level debug --gpus=0,1,2,3 \
--from_checkpoint_path $LOAD_FROM \
--batched-test --not-oldfashioned --disable-log-soft --use-model-config \
--sweep-id 20 --fisher-damp 1e-5 --fisher-subsample-size $FISHER_SUBSAMPLE --fisher-mini-bsz $FISHER_MINI --update-config --prune-class $PRUNER \
--target-sparsity $TARGET --max-mini-bsz ${MAX_MINI_BSZ} \
--full-subsample --fisher-split-grads --fittable-params $FITTABLE \
--woodburry-joint-sparsify --offload-inv --offload-grads \
--result-file $RESULT_CSV --epochs $EPOCHS --eval-fast \
&> $RESULT_PATH

