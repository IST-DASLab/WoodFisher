import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from policies.pruners import GradualPruner
import logging

class FisherPruner(GradualPruner):
    def __init__(self, model, inp_args, **kwargs):
        super(FisherPruner, self).__init__(model, inp_args, **kwargs)

    @staticmethod
    def _get_param_stat(param, param_mask):
        if param is None or param_mask is None: return None
        # Statistic: w_i **2 x F_ii
        # Remember that here param.grad stores avg gradient squared
        # computed during the call to "_compute_avg_sum_grad_squared"
        print("mean value of statistic without eps = {} is ".format(1e-4), torch.mean(param.grad * param ** 2))
        return (param.grad * param ** 2 + 1e-4) * param_mask

    def _release_grads(self):
        optim.SGD(self._model.parameters(), lr=1e-10).zero_grad()

    def _compute_avg_sum_grad_squared(self, dset, subset_inds, device, num_workers):
        self._release_grads()

        tmp_hooks, N = [], len(subset_inds)  #len(dset)
        print("inside fisher where N is ", N)
        for module in self._modules:
            tmp_hooks.append(module.weight.register_hook(lambda grad: grad ** 2 / (2 * N)))
            if module.bias is not None:
                tmp_hooks.append(module.bias.register_hook(lambda grad: grad ** 2 / (2 * N)))

        # batch size is set to 1 as you want to go over each example separately
        dummy_loader = torch.utils.data.DataLoader(dset, batch_size=1, num_workers=num_workers,
                                                   sampler=SubsetRandomSampler(subset_inds))
        for in_tensor, target in dummy_loader:
            in_tensor, target = in_tensor.to(device), target.to(device)
            output = self._model(in_tensor)

            if self.args.disable_log_soft:
                # set to true for resnet20 case
                # set to false for mlpnet as it then returns the log softmax and we go to NLL
                criterion = torch.nn.functional.cross_entropy
            else:
                criterion = F.nll_loss

            loss = criterion(output, target)
            loss.backward()

        for hook in tmp_hooks:
            hook.remove()

    def on_epoch_begin(self, dset, subset_inds, device, num_workers, epoch_num, **kwargs):
        meta = {}
        if self._pruner_not_active(epoch_num):
            return False, {}

        # ensure that the model is not in training mode, this is importance, because
        # otherwise the pruning procedure will interfere and affect the batch-norm statistics
        assert not self._model.training

        self._compute_avg_sum_grad_squared(dset, subset_inds, device, num_workers)
        for module in self._modules:
            level = self._required_sparsity(epoch_num)
            w_stat, b_stat = self._get_param_stat(module.weight, module.weight_mask),\
                             self._get_param_stat(module.bias, module.bias_mask)
            module.weight_mask, module.bias_mask = self._get_pruning_mask(w_stat, level),\
                                                   self._get_pruning_mask(None if self._weight_only else b_stat, level)
        self._release_grads()
        return True, meta