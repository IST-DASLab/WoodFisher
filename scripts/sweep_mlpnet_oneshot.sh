##### instructions ####
# first generate commands via `bash filename> > commands.txt` and then execute `bash commands.txt`
#######################
#!/bin/bash

TARGET_SPARSITYS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.98)
MODULES=("fc1_fc2_fc3")
FISHER_SUBSAMPLE_SIZE="8000"
EPOCH_ENDS=(2)
PRUNERS=(woodtaylor woodfisher magni globalmagni diagfisher)
SEEDS=(0 1 2 3)
PRUNE_EPOCH_FREQ=2
ROOT_DIR="./"
SWEEP_NAME="exp_apr_5"
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
        if [ "${PRUNER}" = "woodtaylor" ]; then
            FISHER_DAMP="1e-1"
        else
            FISHER_DAMP="1e-5"
        fi

        for EPOCH_END in "${EPOCH_ENDS[@]}"
        do
            for SEED in "${SEEDS[@]}"
            do
                for TARGET_SPARSITY in "${TARGET_SPARSITYS[@]}"
                do

                    echo CUDA_VISIBLE_DEVICES=${GPUS[$((${ID} % ${#GPUS[@]}))]} python ${ROOT_DIR}/main.py --exp_name=${SWEEP_NAME} --dset=mnist --dset_path=../datasets --arch=mlpnet --config_path=${ROOT_DIR}/configs/mlpnet_mnist_config_one_shot_woodburry_fisher.yaml --workers=1  --batch_size=64 --logging_level debug --gpus=0 --sweep-id ${ID} --fisher-damp ${FISHER_DAMP} --prune-modules ${MODULE} --fisher-subsample-size ${FISHER_SUBSAMPLE_SIZE} --update-config --prune-class ${PRUNER} --target-sparsity ${TARGET_SPARSITY} --prune-end ${EPOCH_END} --prune-freq ${PRUNE_EPOCH_FREQ} --result-file ${CSV_DIR}/prune_module-${MODULE}_epoch_end-${EPOCH_END}_samples-${FISHER_SUBSAMPLE_SIZE}.csv  --seed ${SEED} --deterministic --full-subsample --woodburry-joint-sparsify '&>' ${LOG_DIR}/${PRUNER}_module-${MODULE}_target_sp-${TARGET_SPARSITY}_epoch_end-${EPOCH_END}_samples-${FISHER_SUBSAMPLE_SIZE}_damp-${FISHER_DAMP}.txt

                    ID=$((ID+1))
                done
            done
        done
    done
done