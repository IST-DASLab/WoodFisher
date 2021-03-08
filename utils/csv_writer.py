import csv
import os

from itertools import chain
from utils import dotdict
from collections import OrderedDict

def dict_union(*args):
    return dict(chain.from_iterable(d.items() for d in args))


def flatten_dict(dictionary, accumulator=None, parent_key=None, separator="_"):
    if accumulator is None:
        accumulator = {}

    for k, v in dictionary.items():
        k = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(v, OrderedDict):
            v = dict(v)
        if isinstance(v, dict):
            flatten_dict(dictionary=v, accumulator=accumulator, parent_key=k)
            continue

        accumulator[k] = v

    return accumulator

def save_experiment_results(args, config, results, resultsK=None):
    if not hasattr(args, 'result_file') or args.result_file == '':
        csv_dir_name = args.run_dir
        args.result_file = os.path.join(csv_dir_name, 'final_results.csv')
    else:
        csv_dir_name = '/'.join(args.result_file.split('/')[0:-1])
        
    os.makedirs(csv_dir_name, exist_ok=True)

    if os.path.exists(args.result_file):
        add_header = False
    else:
        add_header = True

    with open(args.result_file, mode='a') as csv_file:
        results_task_dic = {}

        results_task_dic["init_acc"] = results[0]
        num_prune_epoch = 0
        idx = 1
        while idx in range(1, len(results)-1):
            results_task_dic['before_prune_{}'.format(num_prune_epoch)] = results[idx]
            results_task_dic['after_prune_{}'.format(num_prune_epoch)] = results[idx+1]
            idx += 2
            num_prune_epoch += 1
        results_task_dic["final_acc"] = results[-1]

        if resultsK is not None:
            results_task_dic["init_accK"] = resultsK[0]
            num_prune_epoch = 0
            idx = 1
            while idx in range(1, len(resultsK) - 1):
                results_task_dic['before_pruneK_{}'.format(num_prune_epoch)] = resultsK[idx]
                results_task_dic['after_pruneK_{}'.format(num_prune_epoch)] = resultsK[idx + 1]
                idx += 2
                num_prune_epoch += 1
            results_task_dic["final_accK"] = resultsK[-1]

        results_task_dic['method'] = args.prune_class
        # best epoch stats
        results_task_dic["best_acc"] = args.best_epoch_acc
        results_task_dic["best_epoch"] = args.best_epoch_idx
        # important params which we check often
        results_task_dic['fisher_subsample_size'] = args.fisher_subsample_size
        results_task_dic['fisher_mini_bsz'] = args.fisher_mini_bsz
        results_task_dic['seed'] = args.seed
        results_task_dic['scale_prune_update'] = args.scale_prune_update
        results_task_dic['fisher_split_grads'] = args.fisher_split_grads
        results_task_dic['fittable_params'] = args.fittable_params

        dump_args = dotdict(vars(args))
        dump_config = flatten_dict(dict(config))
        params_dic = dict_union(dump_args, dump_config)
        results_and_params_dic = dict_union(results_task_dic, params_dic)
        # fieldnames = [list(results_task_dic.keys())+ list(args.keys())]
        writer = csv.DictWriter(csv_file, fieldnames=results_and_params_dic.keys())

        if add_header:
            writer.writeheader()
        writer.writerow(results_and_params_dic)
        # writer.writerow({'emp_name': 'Erica Meyers', 'dept': 'IT', 'birth_month': 'March'})

def split_dict_pairs(dic, other_key_nick='numel_'):
    first_dic = {}
    second_dic = {}

    for key, val in dic.items():
        assert len(val) == 3
        first_dic[key] = val[0]
        second_dic[other_key_nick + key] = val[1:]

    return first_dic, second_dic

def save_epoch_results(args, config, results, epoch, sparsity_dic, is_pruning_epoch, resultsK=None):

    csv_dir_name = args.run_dir
    os.makedirs(csv_dir_name, exist_ok=True)

    csv_file_path = os.path.join(csv_dir_name, 'epochwise_results.csv')

    if os.path.exists(csv_file_path):
        add_header = False
    else:
        add_header = True

    with open(csv_file_path, mode='a') as csv_file:
        results_task_dic = {}

        results_task_dic["epoch"] = epoch

        results_task_dic['before_prune'] = results[0]
        results_task_dic['after_prune'] = results[1]
        results_task_dic["final_acc"] = results[2]

        if resultsK is not None:
            results_task_dic['before_pruneK'] = resultsK[0]
            results_task_dic['after_pruneK'] = resultsK[1]
            results_task_dic["final_accK"] = resultsK[2]

        results_task_dic["is_prune"] = is_pruning_epoch

        # save the sparsities
        results_sparsity_dic, results_num_params_dic = split_dict_pairs(sparsity_dic)
        results_task_dic = dict_union(results_task_dic, results_sparsity_dic)
        results_task_dic = dict_union(results_task_dic, results_num_params_dic)

        # save the args and config as usual
        dump_args = dotdict(vars(args))
        dump_config = flatten_dict(dict(config))
        params_dic = dict_union(dump_args, dump_config)
        results_and_params_dic = dict_union(results_task_dic, params_dic)
        # fieldnames = [list(results_task_dic.keys())+ list(args.keys())]
        writer = csv.DictWriter(csv_file, fieldnames=results_and_params_dic.keys())

        if add_header:
            writer.writeheader()
        writer.writerow(results_and_params_dic)
