##### instructions ####
# first generate commands via `bash filename> > commands.txt` and then execute `bash commands.txt`
#######################
#!/bin/bash

TARGET_SPARSITYS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.98)
MODULES=("fc1_fc2_fc3")
FISHER_DAMP=1e-5
EPOCH_END=25
EPOCH_FREQ=25
PRUNERS=(kfac woodfisherblock globalmagni)
JOINTS=(1)
ROOT_DIR="./"
SWEEP_NAME="exp_kfac_comparison"
NOWDATE=""
DQT='"'
GPUS=(0 1 2 3 4 5 6 7)
LOG_DIR="${ROOT_DIR}/${SWEEP_NAME}/log/"
CSV_DIR="${ROOT_DIR}/${SWEEP_NAME}/csv/"
mkdir -p ${LOG_DIR}
mkdir -p ${CSV_DIR}

ID=0
for MODULE in "${MODULES[@]}"
do
    for PRUNER in "${PRUNERS[@]}"
    do
        if [ "${PRUNER}" = "kfac" ]; then
            PIS=(1 0)
            FISHER_SUBSAMPLE_SIZE="50000"
            FISHER_MINIBSZ="1"
        else
            PIS=(0)
            FISHER_SUBSAMPLE_SIZE="5000"
            FISHER_MINIBSZ="10"
        fi

        if [ "${PRUNER}" = "globalmagni" ]; then
            TRUES=(0)
            SEEDS=(0)
        else
            TRUES=(0)
            SEEDS=(0 1 2 3)
        fi

        for TRUE in "${TRUES[@]}"
        do
            for PI in "${PIS[@]}"
            do
                for SEED in "${SEEDS[@]}"
                do
                    for TARGET_SPARSITY in "${TARGET_SPARSITYS[@]}"
                    do
                        for JOINT in "${JOINTS[@]}"
                        do
                            if [ "${JOINT}" = "1" ]; then
                                JOINT_FLAG="--woodburry-joint-sparsify"
                            else
                                JOINT_FLAG=""
                            fi

                            if [ "${PI}" = "1" ]; then
                                PI_FLAG="--kfac-pi"
                            else
                                PI_FLAG=""
                            fi

                            if [ "${TRUE}" = "1" ]; then
                                TRUE_FLAG="--true-fisher"
                            else
                                TRUE_FLAG=""
                            fi

                            echo CUDA_VISIBLE_DEVICES=${GPUS[$((${ID} % ${#GPUS[@]}))]} python ${ROOT_DIR}/main.py --exp_name=exp_aug_12_kfac --dset=mnist --dset_path=../datasets --arch=mlpnet --config_path=/nfs/scistore08/alistgrp/ssingh/projects/neural-compression/configs/mlpnet_mnist_config_one_shot_woodburry_fisher.yaml --workers=1 --batch_size=64 --logging_level debug --gpus=0 --sweep-id ${ID} --fisher-damp ${FISHER_DAMP} --prune-modules fc1_fc2_fc3 --fisher-subsample-size ${FISHER_SUBSAMPLE_SIZE} --fisher-mini-bsz ${FISHER_MINIBSZ} --update-config --prune-class ${PRUNER} --target-sparsity ${TARGET_SPARSITY} --prune-end ${EPOCH_END} --prune-freq ${EPOCH_FREQ} --result-file ${CSV_DIR}/prune_module-${MODULE}_epoch_end-${EPOCH_END}_joint-${JOINT}_true-${TRUE}_nips_5000-10.csv --seed ${SEED} --deterministic --full-subsample --one-shot --from_checkpoint_path checkpoints/mnist_25_epoch_93.97.ckpt --not-oldfashioned --use-model-config ${JOINT_FLAG} ${TRUE_FLAG} ${PI_FLAG} '&>' ${LOG_DIR}/${PRUNER}_module-${MODULE}_target_sp-${TARGET_SPARSITY}_epoch_end-${EPOCH_END}_samples-${FISHER_SUBSAMPLE_SIZE}_${FISHER_MINIBSZ}_joint-${JOINT}_true-${TRUE}_pi-${PI}_damp-${FISHER_DAMP}_seed-${SEED}_sweep-${ID}_5000-10.txt

                            ID=$((ID+1))
                        done
                    done
                done
            done
        done
    done
done