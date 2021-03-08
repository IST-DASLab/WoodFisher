import torch
import torch.nn.functional as F
import copy
import os
import math
from utils.utils import top_accuracy
from utils.datasets import get_datasets
from torch.utils.data import DataLoader
import numpy as np
import logging

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
cmap = plt.cm.Spectral
mpl.style.use('seaborn')


def my_test(args, model, log_dict=None, test_loader=None):
    local_model = copy.deepcopy(model)
    if test_loader is None:
        import torchvision
        test_loader= torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root=args.dset_path, train=False, download=False,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           # only 1 channel
                                           (0.1307,), (0.3081,))
                                   ])),
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )


    #print this gives the same test as the encaaapsulated one!
    if log_dict is None:
        log_dict = {}
        log_dict['test_losses'] = []

    local_model.to(args.device)
    local_model.eval()
    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0

    # but you didn't really use that dataset
    for data, target in test_loader:
        data = data.to(args.device)
        target = target.to(args.device)
        output = local_model(data)
        if args.disable_log_soft:
            test_loss += F.cross_entropy(output, target, size_average=False).item()
        else:
            test_loss += F.nll_loss(output, target, size_average=False).item()

        if not args.report_top5:
            pred = output.data.max(1, keepdim=True)[1]
            correct_top1 += pred.eq(target.data.view_as(pred)).sum()
        else:
            acc_top1, acc_top5 = top_accuracy(args, output, target)
            correct_top1 += acc_top1
            correct_top5 += acc_top5

    logging.info(f"size of test_loader dataset: {len(test_loader.dataset)}")
    test_loss /= len(test_loader.dataset)
    log_dict['test_losses'].append(test_loss)
    logging.info('\nTest set: Avg. loss: {:.6f}, Accuracy: {}/{} ({:.2f}%), top_5: ({:.2f}%)\n'.format(
        test_loss, correct_top1, len(test_loader.dataset),
        float(100. * correct_top1) / len(test_loader.dataset), float(100. * correct_top5) / len(test_loader.dataset)))

    del local_model
    torch.cuda.empty_cache()

    return (float(correct_top1) * 100.0) / len(test_loader.dataset), test_loss, (float(correct_top5) * 100.0) / len(test_loader.dataset)

def my_test_dataset(args, model, test_set, log_dict=None, batched=False, mode='test', device=None):
    local_model = copy.deepcopy(model)
    # this is working fine!
    # mask_layer_name = f"module.{args.prune_modules}._weight_mask"
    # print(f'mask in my_test_dataset is {local_model.state_dict()[mask_layer_name]}')

    #print this gives the same test as the encaaapsulated one!
    if log_dict is None:
        log_dict = {}
        log_dict[f'{mode}_losses'] = []

    # if args.eval_dataparallel:
    #     local_model = torch.nn.DataParallel(local_model)
    #
    # local_model = local_model.to(device)

    if args.always_eval_test:
        print('putting in eval mode')
        local_model.eval()
    else:
        if mode == 'test':
            print('putting in eval mode')
            local_model.eval()
        else:
            print('putting in train mode')
            local_model.train()

    if args.test_batch_size is not None:
        bsz = args.test_batch_size
    else:
        bsz = args.batch_size

    test_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    # print(test_set[0])
    # main difference is that I am not using the data loader!!

    # eval_loss += F.cross_entropy(output, target, reduction='sum').item()
    # pred = output.argmax(dim=1, keepdim=True)

    if not batched:
        for data, target in test_set:
            # print(data, target)
            if 'resnet' in args.arch or args.arch == 'mobilenet' or args.arch == 'simplenet':
                data = data.unsqueeze(0)
            target = torch.Tensor([target]).long()
            # print(data.shape, target.shape)
            data = data.to(device)
            target = target.to(device)
            output = local_model(data)
            if args.disable_log_soft:
                test_loss += F.cross_entropy(output, target, size_average=False).item()
            else:
                test_loss += F.nll_loss(output, target, size_average=False).item()

            if not args.report_top5:
                pred = output.data.max(1, keepdim=True)[1]
                correct_top1 += pred.eq(target.data.view_as(pred)).sum()
            else:
                acc_top1, acc_top5 = top_accuracy(args, output, target)
                correct_top1 += acc_top1
                correct_top5 += acc_top5

        logging.info(f"size of {mode}_loader dataset: {len(test_set)}")
        test_loss /= len(test_set)
        log_dict[f'{mode}_losses'].append(test_loss)
        logging.info('\n{} set: Avg. loss: {:.6f}, Accuracy: {}/{} ({:.2f}%), top_5: ({:.2f}%)\n'.format('T'+mode[1:],
            test_loss, correct_top1, len(test_set),
            float(100. * correct_top1) / len(test_set), float(100. * correct_top5) / len(test_set)))

        del local_model
        torch.cuda.empty_cache()
        return (float(correct_top1) * 100.0) / len(test_set), test_loss, (float(correct_top5) * 100.0) / len(test_set)
    else:
        datas, targets = zip(*test_set)
        datas = list(datas)
        targets = list(targets)
        num_batches = int(math.ceil(len(test_set)/bsz))
        # num_passed = 0

        for idx in range(num_batches):
            data = datas[idx * bsz: (idx + 1) * bsz]
            target = targets[idx * bsz: (idx + 1) * bsz]
            # num_passed += len(data)
            # print(data, target)
            data = torch.stack(data)
            # print(target)
            target = torch.Tensor(target).long()
            # print(data.shape, target.shape)
            data = data.to(device)
            target = target.to(device)
            output = local_model(data)
            if args.disable_log_soft:
                test_loss += F.cross_entropy(output, target, size_average=False).item()
            else:
                test_loss += F.nll_loss(output, target, size_average=False).item()

            if not args.report_top5:
                pred = output.data.max(1, keepdim=True)[1]
                correct_top1 += pred.eq(target.data.view_as(pred)).sum()
            else:
                acc_top1, acc_top5 = top_accuracy(args, output, target)
                correct_top1 += acc_top1
                correct_top5 += acc_top5

            # print(f'correct_top1 at {idx} is {correct_top1} and num_passed is {num_passed}')
        logging.info(f"size of {mode}_loader dataset: {len(test_set)}")
        test_loss /= len(test_set)
        log_dict[f'{mode}_losses'].append(test_loss)
        logging.info('\n{} set: Avg. loss: {:.6f}, Accuracy: {}/{} ({:.2f}%), top_5: ({:.2f}%)\n'.format('T' + mode[1:],
            test_loss, correct_top1, len(test_set),
            float(100. * correct_top1) / len(test_set), float(100. * correct_top5) / len(test_set)))

        del local_model
        torch.cuda.empty_cache()
        return (float(correct_top1) * 100.0) / len(test_set), test_loss, (float(correct_top5) * 100.0) / len(test_set)

def save_plot(args, pts_list, losses_list, nicks, ignore_test=False):

    assert  len(pts_list) == len(losses_list)
    print("this is the losses list", losses_list)
    line_colors = cmap(np.linspace(0, 1, len(nicks)))
    line_color_idx = 0

    # min_val = 1e10
    # max_val = -1
    rootpath = './'
    figpath = os.path.join(rootpath, args.exp_name, 'figures', 'loss_analysis')
    os.makedirs(figpath, exist_ok=True)
    if ignore_test:
        fig, axes = plt.subplots(nrows=1, ncols=1)
        alternate_colors = [ 'xkcd:bordeaux',  'xkcd:cornflower', 'xkcd:tangerine']
    else:
        fig, axes = plt.subplots(nrows=1, ncols=len(losses_list))

    for idx in range(len(losses_list)):
        if idx == 0 and ignore_test:
            continue
        max_ele = float(pts_list[idx][-1])
        pts_list[idx] = [x / max_ele  for x in pts_list[idx]]
        if ignore_test:
            colo = alternate_colors[idx]
            ax = axes
        else:
            ax = axes[idx]
            colo = line_colors[line_color_idx]

        ax.plot(pts_list[idx], losses_list[idx], label=nicks[idx], c=colo, dash_capstyle='round',
             marker='^', markersize=6)
        line_color_idx +=1
        # min_val = min(min_val, min(losses_list[idx]))
        # max_val = max(max_val, max(losses_list[idx]))

        ax.set_xlabel('step size', labelpad=10)
        if ignore_test:
            ax.set_ylabel(f'loss', labelpad=10)
        else:
            ax.set_ylabel(f'{nicks[idx]}_loss', labelpad=10)
        # ax.legend()
        # axes[idx].ylim(bottom=max(min_val - 1, 0), top=min(max_val + 1, 100))

    plt.legend()
    # plt.suptitle('approximation analysis', fontsize=14, y=1.1)
    plt.title('approximation analysis', fontsize=14, y=1.1)

    plt.tight_layout()
    print('save at ', os.path.join(figpath, f'curve_nicks_test_train_{args.prune_class}_{args.prune_modules}_{args.fisher_subsample_size}_epoch-{args.prune_end}_sp-{args.target_sparsity}_zap-{args.zero_after_prune}_pm-{args.previous_mask}_seed-{args.seed}_{args.run_id}.pdf'))

    plt.savefig(os.path.join(figpath, f'curve_nicks_test_train_{args.prune_class}_{args.prune_modules}_{args.fisher_subsample_size}_epoch-{args.prune_end}_sp-{args.target_sparsity}_zap-{args.zero_after_prune}_pm-{args.previous_mask}_seed-{args.seed}_{args.run_id}.pdf'), format='pdf', dpi=1000, quality=95)
    plt.clf()

def compare_against_base(base_model, other_model):
    # assert len(base_model.named_modules()) == len(other_model.named_modules())
    base_dict = dict(base_model.named_modules())
    other_dict = dict(other_model.named_modules())
    # print(base_dict.keys())
    print(base_model.state_dict().keys())
    for name, param in base_model.state_dict().items():
        if isinstance(param, torch.Tensor):
            print("For param {}: are values in base and other model close enough? {}".format(
                name, torch.isclose(param, other_model.state_dict()[name]).all()
            ))
            if 'bn' in name:
                print(param, other_model.state_dict()[name])
        else:
            print(f'no tensor named {name}, continue')
            continue


def compare_models(base_model, other_models):
    for idx, model in enumerate(other_models):
        print(f"For other model at idx: {idx}")
        compare_against_base(base_model, model)

def _init_fn(worker_id):
    np.random.seed(0 + worker_id)

def get_dataloaders(args, data):
    if hasattr(args, 'num_workers'):
        num_workers = args.num_workers
    elif hasattr(args, 'workers'):
        num_workers = args.workers
    else:
        raise NotImplementedError

    data_train, data_test = get_datasets(*data, use_aa=args.aa,
                                         train_random_transforms=not args.disable_train_random_transforms)
    train_loader = DataLoader(data_train, batch_size=args.batch_size,
                              shuffle=not args.disable_train_shuffle, num_workers=num_workers,
                              worker_init_fn=_init_fn)
    test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
                             worker_init_fn=_init_fn)
    return train_loader, test_loader

def get_linear_conv_module_names(model):
    all_module_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            all_module_names.append(name)

    all_module_names = [name.replace('._layer', '') for name in all_module_names]
    return all_module_names

def get_linear_conv_modules(model):
    _module_names = get_linear_conv_module_names(model)
    modules_dict = dict(model.named_modules())

    _modules = [
        modules_dict[module_name] for module_name in _module_names
    ]
    return _modules, _module_names


def analyse_loss_path(self, model, pruning_direction, original_param, mask_previous, mask_overall, datasets,
                      modes, backup_model=None, quad_term=None):

    assert len(datasets) ==  len(modes)
    assert len(self.args.prune_modules.split('_')) == 1
    if isinstance(pruning_direction, list):
        pruning_direction = pruning_direction[0]
    if isinstance(original_param, list):
        original_param = original_param[0]
    if isinstance(mask_previous, list):
        mask_previous = mask_previous[0]
    if isinstance(mask_overall, list):
        mask_overall = mask_overall[0]
    if isinstance(quad_term, list):
        quad_term = quad_term[0]

    if self.args.compare_models and backup_model is not None:
        compare_models(backup_model, [model])

    if isinstance(model, torch.nn.DataParallel):
        prefix = 'module.'
    else:
        prefix = ''
    prune_layer_name = f"{prefix}{self.args.prune_modules}._layer.weight"
    mask_layer_name = f"{prefix}{self.args.prune_modules}._weight_mask"
    # I think if not a DataParllel instance then it shouldn't be prefixed with a 'module'
    print('-----------------\n')

    model_dict = model.state_dict()
    modules_dict = dict(model.named_modules())
    assert len(self.pruners) == 1
    modules = [
        modules_dict[module_name] for module_name in self.pruners[0]._module_names
    ]
    assert len(modules) == 1

    current_mask = modules[0].weight_mask
    print("is equal to mask_overall ", torch.isclose(current_mask, mask_overall).all()) # is True!
    print("is equal to original_param ", torch.isclose(model_dict[prune_layer_name], original_param).all()) # is True!

    if self.args.previous_mask:
        print('mask is before', model_dict[mask_layer_name])
        # all ones, if you are sparsifying for the first time
        print('mask previous is', mask_previous)
        model_dict[mask_layer_name] = mask_previous
        print('mask is now ', model_dict[mask_layer_name])
        # modules[0].weight_mask(mask_previous)
        print('mask is now after access via modules', model_dict[mask_layer_name])

    losses_list = []
    if self.args.local_quadratic:
        quad_losses_list = []
    pruned_params = mask_previous > mask_overall

    print('the value of weights corresponding to those pruned is ', original_param[pruned_params]) # init point
    print('the value of weights corresponding to those in model currently is', model_dict[prune_layer_name][pruned_params]) # dest point
    # ideal dest point would be 0's for those which are pruned (aka pruned_params)
    # the below is just a test of reaching that!
    print('the value of weights corresponding to those in model after adding update is',
          (original_param + pruning_direction)[pruned_params])
    print('the values in pruning update', pruning_direction[pruned_params])
    quad_base_loss = None
    for idx in range(len(datasets)):
        print('Analysing loss in mode ', modes[idx])
        pts = []
        losses = []
        for pt in range(0, int(self.args.num_path_steps) + 1):
            point_weight = ((1.0 * pt)/self.args.num_path_steps)
            print('Point weight is  ', point_weight)

            with torch.no_grad():
                model_dict[prune_layer_name] = original_param + (point_weight * pruning_direction)
                model_dict[mask_layer_name] = mask_previous
            model.load_state_dict(model_dict)
            _, loss, _ = my_test_dataset(self.args, model, datasets[idx], batched=True, mode=modes[idx], device=self.device)
            pts.append(pt)
            losses.append(loss)
            if modes[idx] == 'train' and self.args.local_quadratic:
                if pt == 0:
                    quad_base_loss = loss
                    print('quad_base_loss is ', quad_base_loss)
                    quad_losses_list.append(quad_base_loss)
                else:
                    assert quad_term is not None
                    quad_losses_list.append(quad_base_loss + (point_weight * point_weight * quad_term).item())
                    print("quad model predicts", quad_losses_list[-1])
            if self.args.compare_models and idx == 0 and pt == 0 and backup_model is not None:
                print('comparing after replacing by original param')
                compare_models(backup_model, [model])
        print('-----------------\n')
        losses_list.append(losses)

    if self.args.local_quadratic:
        losses_list.append(quad_losses_list)
        modes.append('local_quadratic_model')
        save_plot(self.args, [pts, pts, pts], losses_list, modes, ignore_test=True)
    else:
        save_plot(self.args, [pts, pts], losses_list, modes)

    # restore weights and masks
    model_dict[prune_layer_name] = original_param
    if self.args.previous_mask:
        model_dict[mask_layer_name] = mask_overall