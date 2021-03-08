"""
This module implements training policies.
For most usecases, only one trainer instance is needed for training and pruning
with a single model. Several trainers can be used for training with knowledge distillation.
"""

import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *
import torch.nn.functional as F
import logging

from policies.policy import PolicyBase

def _accuracy(args, output, target, is_top5=False):
    """Computes the precision@k for the specified values of k"""
    res = []
    if (args is not None and args.report_top5) or is_top5:
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

def build_optimizer_from_config(model, optimizer_config):
    optimizer_class = optimizer_config['class']
    optimizer_args = {k: v for k, v in optimizer_config.items() if k != 'class'}
    optimizer_args['params'] = model.parameters()
    optimizer = globals()[optimizer_class](**optimizer_args)
    return optimizer


def build_lr_scheduler_from_config(optimizer, lr_scheduler_config):
    lr_scheduler_class = lr_scheduler_config['class']
    lr_scheduler_args = {k: v for k, v in lr_scheduler_config.items() if k != 'class'}
    lr_scheduler_args['optimizer'] = optimizer
    epochs = lr_scheduler_args['epochs']
    lr_scheduler_args.pop('epochs')
    lr_scheduler = globals()[lr_scheduler_class](**lr_scheduler_args)
    return lr_scheduler, epochs


def build_training_policy_from_config(model, config_dict, trainer_name, label_smoothing=None):
    trainer_dict = config_dict['trainers'][trainer_name]
    optimizer = build_optimizer_from_config(model, trainer_dict['optimizer'])
    lr_scheduler, epochs = build_lr_scheduler_from_config(optimizer, trainer_dict['lr_scheduler'])
    if label_smoothing is None or label_smoothing <= 0:
        training_criterion = F.cross_entropy
    else:
        from utils.utils import LabelSmoothing
        training_criterion = LabelSmoothing(smoothing=label_smoothing)
    return TrainingPolicy(model, optimizer, lr_scheduler, epochs, training_criterion)


class TrainingPolicy(PolicyBase):
    def __init__(self, model, optimizer, lr_scheduler, epochs, training_criterion=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.model = model
        if training_criterion is not None:
            self.training_criterion = training_criterion
        else:
            self.training_criterion = F.cross_entropy

    def eval_model(self, loader, device, epoch_num):
        self.model.eval()
        eval_loss = 0
        correct = 0
        correctK = 0
        with torch.no_grad():
            for in_tensor, target in loader:
                in_tensor, target = in_tensor.to(device), target.to(device)
                output = self.model(in_tensor)
                eval_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                acc1, acck = _accuracy(None, output, target, is_top5=True)
                correctK += acck
        eval_loss /= len(loader.dataset)
        return eval_loss, correct, correctK

    @property
    def optim_lr(self):
        return list(self.optimizer.param_groups)[0]['lr']

    def on_minibatch_begin(self, minibatch, device, loss, **kwargs):
        """
        Loss can be composite, e.g., if we want to add some KD or
        regularization in future
        """
        self.model.train()
        self.optimizer.zero_grad()
        in_tensor, target = minibatch
        in_tensor, target = in_tensor.to(device), target.to(device)
        output = self.model(in_tensor)
        loss += self.training_criterion(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        acc = 1.0 * correct / target.size(0)
        return loss, acc

    def on_parameter_optimization(self, loss, epoch_num, **kwargs):
        loss.backward()
        self.optimizer.step()

    def on_epoch_end(self, dataloader, device, epoch_num, **kwargs):
        start, freq, end = self.epochs
        if (epoch_num - start) % freq == 0 and epoch_num < end + 1 and start - 1 < epoch_num:
            print("doing lr scheduler step at epoch_num ", epoch_num)
            self.lr_scheduler.step()
        return self.eval_model(dataloader, device, epoch_num)

