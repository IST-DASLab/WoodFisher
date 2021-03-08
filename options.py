import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='WoodFisher: model compression')

    # The following are required parameters, the defaults are for formatting-example purposes
    parser.add_argument('--dset', default='imagenet', type=str,
                        help='dataset for the task (default: "imagenet")')
    parser.add_argument('--dset_path', default='/home/dalistar/ILSVRC', type=str,
                        help='path ot the dataset (default: "/home/dalistar/ILSVRC")')
    parser.add_argument('--arch', default='efficientnetb0', type=str,
                        help='NN architecture fo the task (default: "efficientnetb0")')
    parser.add_argument('--config_path', type=str,
                        help='path to config file')
    parser.add_argument('--pretrained', action='store_true',
                        help='use a pretrained model')
    parser.add_argument('--use_butterfly', action='store_true',
                        help='replace all 2D convolutional layers with Butterfly convolutions')
    parser.add_argument('--use_se', action='store_true',
                        help='using se in mixed conv resnet')
    parser.add_argument('--se_ratio', type=float, default=None,
                        help='se ratio for SELayer (default: 0.5)')
    parser.add_argument('--kernel_sizes', type=int, default=3)
    parser.add_argument('--p', type=int, default=3)

    parser.add_argument('--aa', action='store_true', help='use auto-augment for imagenet')

    # Training-related parameters
    parser.add_argument('--epochs', type=int,
                        help='number of epochs to run (default: 20')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--test-batch_size', default=None, type=int,
                        help='mini-batch size for test (default: None)')
    parser.add_argument('--recompute_bn_stats', action='store_true',
                        help='recompute bn statistics after pruning')
    parser.add_argument('--num_samples', default=4096, type=int,
                        help='number of samples to compute pruning statistics for Fisher and SNIP based pruners (default: 4096)')

    # Compute parameters
    parser.add_argument('--workers', default=4, type=int,
                        help='number of workers to load the data (default: 4)')
    parser.add_argument('--cpu', action='store_true',
                        help='force training on CPU')
    parser.add_argument('--gpus', default=None,
                        help='Comma-separated list of GPU device ids to use, this assumes that parallel is applied (default: all devices)')

    # Run history management
    # The most convenient way is to specify --exp_name, then the logs and models will be stored under
    # ../exp_root/{exp_name}/{current_inferred_datetime}/
    parser.add_argument('--experiment_root_path', type=str, default='../exp_root',
                        help='path to directory under which all experiments will be stored; you can leave this argument as is')
    parser.add_argument('--exp_name', type=str, default='default_exp',
                        help='name of the experiment, will be used to name a subdirectory of experiments_root;' +
                             'in this subdirectory, all runs (named by datetime) of this experiment will be stored')
    parser.add_argument('--logging_level', type=str, default='info',
                        help='logging level: debug, info, warning, error, critical (default: info)')
    parser.add_argument('--training_stats_freq', type=int, default=30,
                        help='the frequency (number of minibatches) to track training stats, e.g., loss, accuracy etc. (default: 30)')

    parser.add_argument('--checkpoint_freq', type=int, default=1,
                        help='epoch frequency with which the checkpoints are dumped; at each time, two checkpoints are maintained:' +
                             'latest and best on validation/test set')
    parser.add_argument('--from_checkpoint_path', type=str, default=None,
                        help='specifies path to *run_dir* from which the progress should be resumed')
    parser.add_argument('--use-model-config', action='store_true',
                        help='use current model config for checkpoint too! used in case ckpt was not saved in the right way')
    parser.add_argument('--reset_training_policy', action='store_true',
                        help='if reset training policy optimizer and lr_scheduler to initial config')
    parser.add_argument('--load_distiller_weights_from', type=str, default=None,
                        help='the path to download the weight to the model from distiller checkpoint')
    parser.add_argument('--ckpt-epoch', default=-1, type=int, help='at what epoch was the checkpoint taken')

    # MLPNet specific args
    parser.add_argument('--num-hidden-nodes1', default=40, type=int,
                        help='mlpnet: number of hidden nodes in the hidden layer 1')
    parser.add_argument('--num-hidden-nodes2', default=20, type=int,
                        help='mlpnet: number of hidden nodes in the hidden layer 2')
    parser.add_argument('--num-classes', default=10, type=int,
                        help='number of target classes')
    parser.add_argument('--disable-log-soft', action='store_true', help='disable log softmax for mlpnet')

    parser.add_argument('--seed', default=0, type=int, help='seed the computations!')
    parser.add_argument('--enable-dropout', action='store_true', help='enable dropout for MLPNet')
    parser.add_argument('--disable_bias', action='store_false', help='disable bias in the neural network layers')

    # fisher related

    parser.add_argument('--fisher-seed', default=-1, type=int, help='seed the compute_emprical_fisher method')
    parser.add_argument('--fisher-parts', default=5, type=int, help='num of parts to divide fisher computation into')
    parser.add_argument('--fisher-optimized', action='store_true',
                        help='split the addition of outer products into parts to save memory')
    parser.add_argument('--save-fisher', action='store_true', help='to dump the emp_fisher to storage or not')
    parser.add_argument('--fisher-cpu', action='store_true', help='compute the fisher on cpu!!')
    parser.add_argument('--load-fisher', default="", type=str, help='path from where to load the emp_fisher')
    parser.add_argument('--fisher-subsample-size', type=int, default=32, action='store',
                        help='over what number of training data is the empirical fisher computed')
    parser.add_argument('--fisher-damp', type=float, default=1e-3, action='store',
                        help='dampening factor to scale the identity matrix and make fisher invertible')
    parser.add_argument('--aux-gpu-id', default=-1, type=int, help='GPU id to use')


    parser.add_argument('--update-config', action='store_true',
                        help='update the config, all these below will affect only when this flag is enabled')
    parser.add_argument('--prune-class', type=str, default='woodfisher',
                        choices=['magni', 'globalmagni', 'naivehess', 'diagfisher',
                                 'woodfisher', 'woodtaylor', 'woodfisherblock', 'woodtaylorblock',
                                 'kfac', 'woodfisherblockgroup', 'woodfisherblockdynamic', 'woodfisherblock_flops'],
                        help='which pruner to use (assumes only 1 pruner)')

    parser.add_argument('--prune-optimizer', type=str, default=None, choices=['Adam', 'SGD', 'RMSprop'],
                        help='optimizer to use during retrain while pruning')

    parser.add_argument('--prune-start', type=int, default=None,
                        help='starting epoch for gradual pruning procedure')
    parser.add_argument('--prune-end', type=int, default=None,
                        help='ending epoch for gradual pruning procedure')
    parser.add_argument('--prune-freq', type=int, default=None,
                        help='frequency at which pruning is carried out!')
    parser.add_argument('--prune-modules', type=str, default=None,
                        help='which modules to prune: space separated string with module names')
    parser.add_argument('--untrained-last', action='store_true', help='dont prune after the last epoch!')
    parser.add_argument('--init-sparsity', action='store', type=float, default=None,
                        help='initial sparsity')
    parser.add_argument('--target-sparsity', action='store', type=float, default=None,
                        help='target sparsity to achieve!')
    parser.add_argument('--one-shot', action='store_true',
                        help='one-shot pruning after training for prune_start-prune_end epochs!')
    parser.add_argument('--prune-bias', action='store_true', default=False, help='prune biases as well')
    parser.add_argument('--prune-all', action='store_true', default=False, help='prune all modules in the network')
    parser.add_argument('--prune-lr', default=None, type=float,
                        help='learning rate for retraining part in pruning')
    parser.add_argument('--prune-momentum', default=None, type=float,
                        help='momentum for retraining part in pruning')
    parser.add_argument('--set-prune-momentum', action='store_true',
                        help='add momentum to config file if not already set there')
    parser.add_argument('--prune-wdecay', default=None, type=float, help='weight decay for retraining part in pruning')
    parser.add_argument('--result-file', action='store', type=str, default='',
                        help='path to file containing the saved results')
    parser.add_argument('--sweep-id', default=-1, type=int, help='id of the experiment in sweep')

    parser.add_argument('--not-oldfashioned', action='store_true', help='the checkpoints are not old_fashioned!')
    parser.add_argument('--ignore-prefix', action='store_true', help='ignore the module prefix used!')
    parser.add_argument('--batched-test', action='store_true', help='custom test dataset in batched mode!')
    parser.add_argument('--cache-subset-ids', action='store_true',
                        help='sample subset indices only once in a hope to reduce the randomness!')
    parser.add_argument('--full-subsample', action='store_true', help='do full subsampling in every epoch!')
    parser.add_argument('--deterministic', action='store_true', help='makes things deterministic!')
    parser.add_argument('--normalize-hgp', action='store_true',
                        help='normalize the hessain gradient product term for update in woodtaylor!')
    parser.add_argument('--prune-direction', action='store_true', help='get the pruning direction for loss analysis')
    parser.add_argument('--num-path-steps', default=-1, type=float, action='store',
                        help='num of discretization steps for loss path analysis')
    parser.add_argument('--previous-mask', action='store_true', help='use the mask before pruning for loss analysis')
    parser.add_argument('--zero-after-prune', action='store_true',
                        help='set update so that it is zero after prune (for woodburry like)')
    parser.add_argument('--compare-models', action='store_true', help='compare models with one copied before')
    parser.add_argument('--no-dataparallel', action='store_true', help='dont use dataparallel')
    parser.add_argument('--always-eval-test', action='store_true',
                        help='always test in eval mode (even when evaluating train loss)')
    parser.add_argument('--disable-train-random-transforms', action='store_true',
                        help='disable random transforms in train dataset ')
    parser.add_argument('--disable-train-shuffle', action='store_true', help='disable shuffling in train dataloader')
    parser.add_argument('--check-train-loss', action='store_true', help='if check train loss at various places')
    parser.add_argument('--save-before-prune-ckpt', action='store_true', help='save checkpoint just before pruning!')
    parser.add_argument('--inspect-inv', action='store_true', help='inspect the inverse!')
    parser.add_argument('--fisher-mini-bsz', default=1, type=int, action='store',
                        help='minibatch of gradients to avg whose outer product then takes place!')
    parser.add_argument('--max-mini-bsz', default=None, type=int, action='store',
                        help='max size of mini bsz that can fit in a machine. If fisher-mini-bsz greater than this, make the batches of this size')
    parser.add_argument('--prune-at-launch', action='store_true',
                        help='start pruning at the very first epoch when pretraining!')
    parser.add_argument('--layer-trace-stat', action='store_true',
                        help='use layer trace stats and multiply by weight_stats computed in layer!')
    parser.add_argument('--woodburry-joint-sparsify', action='store_true',
                        help='jointly compute param stats in woodburry!')
    parser.add_argument('--dump-grads-mat', action='store_true',
                        help='dump grads in matlab format!')
    parser.add_argument('--dump-fisher-inv-mat', action='store_true',
                        help='dump fisher inverse computed via woodfisher in matlab format!')
    parser.add_argument('--fisher-trace', action='store_true',
                        help='compute layerwise traces of empirical fisher matrix')
    parser.add_argument('--eps', default=1e-10, type=float, action='store',
                        help='constant added to prevent divide by 0')
    parser.add_argument('--check-grads', action='store_true',
                        help='check if the grads based on which jl is computed is same as current grads')
    parser.add_argument('--true-fisher', action='store_true',
                        help='use true fisher (sample y from model) rather than empirical fisher ')

    parser.add_argument('--fisher-split-grads', action='store_true',
                        help='split the grads to fit the outer product in memory!')
    parser.add_argument('--offload-inv', action='store_true',
                        help='offload block inverses to CPU for woodburry joint sparsify efficient!')
    parser.add_argument('--fittable-params', action='store', default=-1, type=int,
                        help='number of parameters which woodfisher can accommodate in GPU memory!')
    parser.add_argument('--offload-grads', action='store_true',
                        help='offload the grads collected to prevent oom!')
    parser.add_argument('--eval-fast', action='store_true',
                        help='do eval faster my_test methods instead of my_test_dataset')
    parser.add_argument('--export_onnx', action='store_true', help='export onnx')
    parser.add_argument('--fisher-damp-correction', action='store_true',
                        help='fix the incorrect division in the 1st iterate numerator')
    parser.add_argument('--grad-subsample-size', type=int, default=None, action='store',
                        help='over what number of training data is the full gradient computed for taylor series')
    parser.add_argument('--normalize-update', action='store_true', help='normalize the weight update for woodtaylor!')
    parser.add_argument('--normalize-update-mult', type=float, default=1, action='store',
                        help='multiplier while normalizing the weight update for woodtaylor!')
    parser.add_argument('--kfac-pi', action='store_true', help='use pi based dampening for KFAC!')
    parser.add_argument('--local-quadratic', action='store_true',
                        help='analyse the training loss given by local quadratci model')
    parser.add_argument('--compare-globalmagni-mask', action='store_true',
                        help='compare the layerwise mask generated via (joint) WF with global magni')
    parser.add_argument('--spearman-globalmagni', action='store_true',
                        help='rank correlation between the pruning statistic from global magni and WF')
    # UPDATE: both the --subtract-min and --check-reintro are not an issue
    # as the diagonal of the inverse fisher will always be positive.
    parser.add_argument('--subtract-min', action='store_true',
                        help='subtract the minimum of the statistic for WF')

    parser.add_argument('--repeated-one-shot', action='store_true',
                        help='Repeated one-shot pruning after training, and with no fine-tuning!')
    parser.add_argument('--scale-prune-update', type=float, default=1, action='store',
                        help='multiplier for reducing the effect of callibrating the other weights!')
    parser.add_argument('--centered', action='store_true',
                        help='center the empirical fisher matrix (currently supported in WoodTaylor)!')
    parser.add_argument('--flops', action='store_true', help='get flops')
    parser.add_argument('--flops-power', type=float, default=0, help='exponent of the flop based statistic which gets multiplied to form overall stat')
    parser.add_argument('--flops-per-param', action='store_true', help='normalize by the param count of respective layer')
    parser.add_argument('--flops-normalize', type=str, default=None, help='balance out the flop counts')
    parser.add_argument('--flops-target', type=float, default=-1, help='precise target of FLOPs to achieve')
    parser.add_argument('--flops-epsilon', type=float, default=1, help='tolerance in achieving the target FLOP count')

    parser.add_argument('--topk', action='store_true', help='print topK accuracy')
    parser.add_argument('--mask-onnx', action='store_true', help='apply masks for onnx checkpoint')
    parser.add_argument('--onnx-nick', type=str, default=None, action='store', help='name for generated onnx')
    parser.add_argument('--save-dense-also', action='store_true',
                        help='also save the dense model by applying masks to sparsified model')

    parser.add_argument('--recompute-schedule', type=str, default=None, action='store', choices=["linear", "poly"],
                        help='type of schedule for recomputation')
    parser.add_argument('--recompute-num', type=int, default=None, help='number of recompute steps to carry out')
    parser.add_argument('--recompute-degree', type=float, default=None, help='degree of the polynomial recompute steps')

    parser.add_argument('--disable-wdecay-after-prune', action='store_true', help='Disables the weight decay after prune (sets to zero)')
    parser.add_argument('--woodtaylor-abs', action='store_true', help='consider absolute value in woodtaylor stats')
    parser.add_argument('--fisher-effective-damp', action='store_true', help='Use the effective dampening constant, i.e., WF (positive definite) + training')

    # Flags which are not used for the primary results, but can be played around with if needed!

    parser.add_argument('--label-smoothing', type=float, default=0, help='how much to soften the labels for training loss')
    parser.add_argument('--hess-label-smoothing', type=float, default=None, help='how much to soften the labels for hessian')

    return parser.parse_args()
