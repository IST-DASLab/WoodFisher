"""
Miscellaneous utilities
"""
import torch
import functools
import inspect
import warnings
import logging
from math import ceil
from utils.masking_utils import is_wrapped_layer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn


def top_accuracy(args, output, target):
    """Computes the precision@k for the specified values of k"""
    res = []
    if args.report_top5:
        topks = (1, 5)
    else:
        topks = (1)

    if len(topks) > 0:
        maxk = max(topks)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for topk in topks:
            correct_k = correct[:topk].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
    else:
        res += [0]
    return res

def get_normal_stats(tensor):
    return tensor.mean(), tensor.std()

def get_summary_stats(tensor):
    return {'max': tensor.max().item(),
            'mean': tensor.mean().item(),
            'min': tensor.min().item(),
            'median': tensor.median().item(),
            'std': tensor.std().item()
            }

def normalize_module_name(layer_name):
    """
    Normalize a each module's name in nn.Model
    in case of model was wrapped with DataParallel
    """
    modules = layer_name.split('.')
    try:
        idx = modules.index('module')
    except ValueError:
        return layer_name
    del modules[idx]
    return '.'.join(modules)


class TrainingProgressTracker(object):
    stats = ['loss', 'acc', 'time', 'lr']
    meta_keys = ['hess_diag_negatives', 'bottom magnitudes', 'weights']

    def __init__(self, start_epoch, train_size, val_size, freq, tb_logdir):
        self.train_size = train_size
        self.val_size = val_size
        self.freq = freq
        self.epoch = start_epoch

        self.progress = 0
        self.tb_logger = SummaryWriter(log_dir=tb_logdir)

        self.init_stats()

    @property
    def should_write(self):
        return (
                self.progress != 0 and not (self.progress + 1) % self.freq
                or (self.progress + 1 == self.train_size)
        )

    @property
    def scaling_factor(self):
        if self.progress + 1 > self.train_size - self.train_size % self.freq:
            return self.train_size % self.freq
        return self.freq

    def init_stats(self):
        for stat in TrainingProgressTracker.stats:
            setattr(self, stat, 0.)

    def damp_progress(self, **kwargs):
        for stat in TrainingProgressTracker.stats:
            try:
                new_stat = getattr(self, stat) + kwargs[stat] / self.scaling_factor
                setattr(self, stat, new_stat)
            except:
                raise KeyError(f'Tracking of training statistic {stat} is not implemented ' \
                                   f'the list of allowed statistics {TrainingProgressTracker.stats}')

    def write_progress(self):
        for stat in TrainingProgressTracker.stats:
            self.tb_logger.add_scalar(stat,
                                      getattr(self, stat),
                                      self.epoch * self.train_size + self.progress + 1)

        logging.info(f'Epoch [{self.epoch}] [{self.progress + 1}/{self.train_size}]:    ' +
                     f'Loss: {self.loss:.6f}    ' +
                     f'Top1: {self.acc:.6f}    ' +
                     f'Time: {self.time:.6f}    ' +
                     f'LR: {self.lr:.6f}')

        self.init_stats()

    def step(self, **kwargs):
        self.damp_progress(**kwargs)
        if self.should_write:
            self.write_progress()

        self.progress += 1
        if self.progress > self.train_size - 1:
            self.progress = 0
            self.epoch += 1

    def tensorboard_info(self, epoch_num, val_loss, val_acc_percent):
        self.tb_logger.add_scalar('val_loss', val_loss, epoch_num)
        self.tb_logger.add_scalar('val_acc', val_acc_percent/100.0, epoch_num)

    def val_info(self, epoch_num, val_loss, val_correct):
        self.tb_logger.add_scalar('val_loss', val_loss, epoch_num)
        self.tb_logger.add_scalar('val_acc', 1.0 * val_correct / self.val_size, epoch_num)

        logging.info(f'Epoch [{epoch_num}] Test set: Average loss: {val_loss:.4f}, ' +
                     f'Top1: {val_correct:.0f}/{self.val_size} ' +
                     f'({100. * val_correct / self.val_size:.2f}%)\n'
                     )

    def sparsity_info(self,
                      epoch_num,
                      sparsity_dicts,
                      total_num_zeros,
                      total_num_params):

        logging.info(f'\nEpoch [{epoch_num}]: '
                     'Sparsity information for pruned layers:')
        print(total_num_zeros, total_num_params)
        print(sparsity_dicts)
        processed_sparsity_dict = {}
        processed_sparsity_dict['total'] = (float(total_num_zeros) / total_num_params, total_num_zeros, total_num_params)

        for sparsity_dict in sparsity_dicts:
            for name, (num_zeros, num_params) in sparsity_dict.items():
                self.tb_logger.add_scalar(name + '_sparsity',
                                          float(num_zeros) / (num_params + 1),
                                          epoch_num)

                logging.info(f'\t{name}:\t{float(num_zeros) / (num_params + 1):.4f} '
                             f'(pruned {num_zeros}/{num_params + 1})')
                processed_sparsity_dict[name] = (float(num_zeros) / num_params, num_zeros, num_params)

        logging.info(f'\tTotal sparsity: {float(total_num_zeros) /(total_num_params + 1):.4f} '
                     f'(pruned {total_num_zeros}/{total_num_params + 1}) \n')

        return processed_sparsity_dict

    @staticmethod
    def _key_in_metas(key, metas):
        return any([key in meta for meta in metas])

    def meta_info(self, epoch_num, metas):
        # print(metas)
        for key in TrainingProgressTracker.meta_keys:
            if key == 'bottom magnitudes':
                for meta in metas:
                    if key not in meta:
                        continue
                    info_list = list(meta[key].items())
                    for info in info_list:
                        self.tb_logger.add_scalar('bottom_magnitudes' + info[0],
                                                  info[1].item(), epoch_num)

            if key == 'weights':
                for meta in metas:
                    if key not in meta:
                        continue
                    info_list = list(meta[key].items())
                    for info in info_list:
                        self.tb_logger.add_histogram('weights_of_' + info[0],
                                                     info[1], epoch_num)

            if key == 'hess_diag_negatives':
                if not self._key_in_metas(key, metas):
                    continue
                logging.info(f'\nHessian diag negatives ratio info:')
                total_neg, total = 0, 0
                for meta in metas:
                    if key not in meta:
                        continue
                    info_list = list(meta[key].items())
                    for info in info_list:
                        total_neg, total = total_neg + info[1][0], total + info[1][1]
                        self.tb_logger.add_scalar('hess_diag_neg_ratio_' + info[0],
                                                  info[1][0].float() / info[1][1],
                                                  epoch_num)
                        logging.info(f'\t{info[0]}: {info[1][0].float() / info[1][1]:.6f} '
                                     f'({info[1][0]}/{info[1][1]})')

                self.tb_logger.add_scalar('hess_diag_neg_ratio',
                                          total_neg.float() / total,
                                          epoch_num)
                logging.info(f'\ttotal: {total_neg.float() / total:.6f} '
                             f'({total_neg}/{total})')
            # else:
            #     raise NotImplemented(f'Parsing of {key} meta information')
        logging.info('')


def preprocess_for_device(args):
    if torch.cuda.is_available() and not args.cpu:
        args.device = torch.device('cuda')
        if args.gpus is not None:
            try:
                args.gpus = list(map(int, args.gpus.split(',')))
            except:
                raise ValueError('GPU_ERROR: argument --gpus should be a comma-separated list of integers')
            num_gpus = torch.cuda.device_count()
            if any([gpu_id >= num_gpus for gpu_id in args.gpus]):
                raise ValueError('GPU_ERROR: specified gpu ids are not valid, check that gpu_id <= num_gpus')
            torch.cuda.set_device(args.gpus[0])
    else:
        args.gpus = -1
        args.device = torch.device('cpu')
    return args


def get_total_sparsity(module):
    if hasattr(module, "weight"):
        if is_wrapped_layer(module):
            wz, bz, total = module.layer_sparsity
            return wz + bz, total
        else:
            num_params = module.weight.numel()
            num_zeros = (module.weight.data == 0).sum()
            if hasattr(module, "bias") and module.bias is not None:
                num_params += module.bias.numel()
                num_zeros += (module.bias.data == 0).sum()
            return num_zeros, num_params
    num_zeros, num_params = 0, 0
    for child_module in module.children():
        num_zeros_child, total_child = get_total_sparsity(child_module)
        # print(f"for child_module {child_module}, num_zeros: {num_zeros_child} & total_child: {total_child}")
        num_zeros += num_zeros_child
        num_params += total_child
    return num_zeros, num_params


def get_total_sparsity_unwrapped(module):
    if hasattr(module, "weight"):
        num_sparse = (module.weight.data == 0).float().sum()
        num_params = module.weight.numel()
        if hasattr(module, "bias") and module.bias is not None:
            num_params += module.bias.numel()
            num_sparse += (module.bias.data == 0).float().sum()
        return num_sparse, num_params
    num_zeros, num_params = 0, 0
    for child_module in module.children():
        num_zeros_child, total_child = get_total_sparsity_unwrapped(child_module)
        # print(f"for child_module {child_module}, num_zeros: {num_zeros_child} & total_child: {total_child}")
        num_zeros += num_zeros_child
        num_params += total_child
    return num_zeros, num_params


def recompute_bn_stats(model, dataloader, device):
    """
    Recomputes batch normalization statistics after pruning or
    reintroduction steps of scheduler.

    Arguments:
        module {torch.nn.Module} -- PyTorch module implementing NN
        dataloader {torch.utils.data.DataLoader} -- ddataloader
        device {torch.device or string} - device for (input, target) tensors
    """
    logging.info('Manager is running batch-norm statistics recomputation')
    with torch.no_grad():
        model.train()
        for input, target in tqdm(dataloader):
            input, target = input.to(device), target.to(device)
            output = model(input)


def percentile(tensor, p):
    """
    Returns percentile of tensor elements

    Arguments:
        tensor {torch.Tensor} -- a tensor to compute percentile
        p {float} -- percentile (values in [0,1])
    """
    if p > 1.:
        raise ValueError(f'Percentile parameter p expected to be in [0, 1], found {p:.5f}')
    k = ceil(tensor.numel() * (1 - p))
    if p == 0:
        return -1  # by convention all param_stats >= 0
    # topk returns a tuple: (values, indices) and essentially amongst the k percentile we are returning the smallest value
    return torch.topk(tensor.view(-1), k)[0][-1]

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def flatten_tensor_list(tensors):
    flattened = []
    for tensor in tensors:
        flattened.append(tensor.view(-1))
    return torch.cat(flattened, 0)

def dump_tensor_to_mat(tens, path, name, mat_key, transpose=False):
    import scipy.io
    import os
    if transpose:
        scipy.io.savemat(os.path.join(path, name), dict({mat_key: tens.t().cpu().numpy()}))
    else:
        scipy.io.savemat(os.path.join(path, name), dict({mat_key: tens.t().cpu().numpy()}))

def add_outer_products_efficient_v1(mat, vec, num_parts=2):
    piece = int(ceil(len(vec) / num_parts))
    vec_len = len(vec)
    for i in range(num_parts):
        for j in range(num_parts):
            mat[i * piece:min((i + 1) * piece, vec_len), j * piece:min((j + 1) * piece, vec_len)].add_(
                torch.ger(vec[i * piece:min((i + 1) * piece, vec_len)], vec[j * piece:min((j + 1) * piece, vec_len)])
            )
    return mat

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    From: https://github.com/RAIVNLab/STR/blob/master/utils/net_utils.py
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target, reduction=None):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()