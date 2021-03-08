NOWDATE=$(date '+%Y_%m_%d_%H:%M:%S') # :%N for further granularity
EXP_NAME=test_code_${NOWDATE}
RESULT_FILE=./tests/results_${NOWDATE}.csv

# kfac
echo CUDA_VISIBLE_DEVICES=0 python .//main.py --exp_name=${EXP_NAME} --dset=mnist --dset_path=../datasets --arch=mlpnet --config_path=./configs/mlpnet_mnist_config_one_shot_woodburry_fisher.yaml --workers=1 --batch_size=64 --logging_level debug --gpus=0 --sweep-id 64 --fisher-damp 1e-5 --prune-modules fc1_fc2_fc3 --fisher-subsample-size 100 --fisher-mini-bsz 1 --update-config --prune-class kfac --target-sparsity 0.95 --prune-end 25 --prune-freq 25 --seed 1 --deterministic --full-subsample --one-shot --from_checkpoint_path checkpoints/mnist_25_epoch_93.97.ckpt --not-oldfashioned --use-model-config --woodburry-joint-sparsify --result-file ${RESULT_FILE}

# woodfisherblock
echo CUDA_VISIBLE_DEVICES=1 python .//main.py --exp_name=${EXP_NAME} --dset=mnist --dset_path=../datasets --arch=mlpnet --config_path=./configs/mlpnet_mnist_config_one_shot_woodburry_fisher.yaml --workers=1 --batch_size=64 --logging_level debug --gpus=0 --sweep-id 64 --fisher-damp 1e-5 --prune-modules fc1_fc2_fc3 --fisher-subsample-size 100 --fisher-mini-bsz 1 --update-config --prune-class woodfisherblock --target-sparsity 0.95 --prune-end 25 --prune-freq 25 --seed 1 --deterministic --full-subsample --one-shot --from_checkpoint_path checkpoints/mnist_25_epoch_93.97.ckpt --not-oldfashioned --use-model-config --woodburry-joint-sparsify --result-file ${RESULT_FILE}

# diagfisher
echo CUDA_VISIBLE_DEVICES=2 python .//main.py --exp_name=${EXP_NAME} --dset=mnist --dset_path=../datasets --arch=mlpnet --config_path=./configs/mlpnet_mnist_config_one_shot_woodburry_fisher.yaml --workers=1 --batch_size=64 --logging_level debug --gpus=0 --sweep-id 64 --fisher-damp 1e-5 --prune-modules fc1_fc2_fc3 --fisher-subsample-size 100 --fisher-mini-bsz 1 --update-config --prune-class diagfisher --target-sparsity 0.95 --prune-end 25 --prune-freq 25 --seed 1 --deterministic --full-subsample --one-shot --from_checkpoint_path checkpoints/mnist_25_epoch_93.97.ckpt --not-oldfashioned --use-model-config --woodburry-joint-sparsify --result-file ${RESULT_FILE}
