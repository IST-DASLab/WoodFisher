##### instructions ####
# first generate commands via `bash filename> > commands.txt` and then execute `bash commands.txt`
#######################
#!/bin/bash
TARGET_SPARSITYS=(0.1 0.2 0.3 0.4 0.5 0.6 0.65)
SEEDS=(0 1 2)
FISHER_SUBSAMPLE_SIZES=(400)
FISHER_MINIBSZS=(400)
PRUNERS=(woodfisherblock)
JOINTS=(0)
FITTABLE_PARAMS=(2000)

FISHER_DAMP="1e-5"
ROOT_DIR="."
DATA_DIR="~/ILSVRC/"
SWEEP_NAME="exp_may_25_LOBS_resnet50"
NOWDATE=""
DQT='"'
GPUS=("0,1,2,3")
LOG_DIR="${ROOT_DIR}/exp_neurips/${SWEEP_NAME}/log/"
CSV_DIR="${ROOT_DIR}/exp_neurips/${SWEEP_NAME}/csv/"
mkdir -p ${LOG_DIR}
mkdir -p ${CSV_DIR}

ID=0
for FITTABLE_PARAM in "${FITTABLE_PARAMS[@]}"
do
    for JOINT in "${JOINTS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do
            for TARGET_SPARSITY in "${TARGET_SPARSITYS[@]}"
            do
                for FISHER_MINIBSZ in "${FISHER_MINIBSZS[@]}"
                do
                    for PRUNER in "${PRUNERS[@]}"
                    do
                       for FISHER_SUBSAMPLE_SIZE in "${FISHER_SUBSAMPLE_SIZES[@]}"
                       do

                            echo CUDA_VISIBLE_DEVICES=${GPUS[$((${ID} % ${#GPUS[@]}))]} python ${ROOT_DIR}/main.py --exp_name=${SWEEP_NAME} --dset=imagenet --dset_path=${DATA_DIR} --arch=resnet50 --config_path=${ROOT_DIR}/configs/resnet50.yaml --workers=20 --batch_size=256 --logging_level debug --gpus=0,1,2,3 --pretrained --batched-test --not-oldfashioned --disable-log-soft --use-model-config --sweep-id ${ID} --fisher-damp ${FISHER_DAMP} --fisher-subsample-size ${FISHER_SUBSAMPLE_SIZE} --fisher-mini-bsz ${FISHER_MINIBSZ} --update-config --prune-class ${PRUNER} --target-sparsity ${TARGET_SPARSITY} --result-file ${CSV_DIR}/imagenet_prune_module-all-gradual_joint-${JOINT}_samples-${FISHER_SUBSAMPLE_SIZE}_${FISHER_MINIBSZ}_damp-${FISHER_DAMP}_one-shot_lobs.csv --seed ${SEED} --deterministic --full-subsample --fisher-split-grads --fittable-params ${FITTABLE_PARAM} --offload-inv --offload-grads --one-shot --topk '&>' ${LOG_DIR}/imagenet_woodfisher-joint_module-all_gradual_samples-${FISHER_SUBSAMPLE_SIZE}_${FISHER_MINIBSZ}_damp-${FISHER_DAMP}_seed-${SEED}_spar-${TARGET_SPARSITY}_bsz_256_split_${FITTABLE_PARAM}_offload_inv_grads_lobs.txt

                            ID=$((ID+1))

                        done
                    done
                done
            done
        done
    done
done