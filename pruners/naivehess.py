import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import logging
from utils.utils import flatten_tensor_list, get_summary_stats, dump_tensor_to_mat
from policies.pruners import GradualPruner
import math

class NaiveHessianPruner(GradualPruner):
    '''
    Brute force computation of the diagonal of the Hessian to get estimates for pruning
    '''
    def __init__(self, model, inp_args, **kwargs):
        super(NaiveHessianPruner, self).__init__(model, inp_args, **kwargs)

    @staticmethod
    def _get_param_stat(param, param_mask):
        if param is None or param_mask is None: return None
        # statistic can be negative so zeros breaking sparsity level
        # can substract (minimal + eps) and then zero out pruned stats
        param_stat = param.pow(2).mul(param.hess_diag)
        return (param_stat - param_stat.min() + 1e-8) * param_mask

    def _release_grads(self):
        optim.SGD(self._model.parameters(), lr=1e-10).zero_grad()

    def _add_hess_attr(self):
        self._release_grads()
        for param in self._model.parameters():
            setattr(param, 'hess_diag', torch.zeros_like(param))

    def _del_hess_attr(self):
        self._release_grads()
        for param in self._model.parameters():
            delattr(param, 'hess_diag')

    def _compute_second_derivatives(self):
        for module in self._modules:
            for param in module.parameters():
                basis = torch.zeros_like(param.flatten())
                tmp_diag_hess = torch.zeros_like(param.flatten())
                for i in tqdm(range(param.numel())):
                    basis[i] = 1
                    tmp_diag_hess[i] += (torch.autograd.grad(param.grad, param, grad_outputs=basis.view_as(param),
                                                             retain_graph=True)[0]).flatten()[i]
                    basis[i] = 0
                param.hess_diag += tmp_diag_hess.view(param.shape)

        if self.args.dump_hess_mat:
            # extra functionality to dump hessian into matlab format for later analysis
            diag_hess = []
            for module in self._modules:
                for param in module.parameters():
                    if hasattr(param, 'hess_diag') and param.hess_diag is not None:
                        diag_hess.append(param.hess_diag.flatten())
            diag_hess = flatten_tensor_list(diag_hess)
            dump_tensor_to_mat(diag_hess, self.args.run_dir, 'diag_hess.mat', 'diag_hess')

    def _compute_diag_hessian(self, dset, subset_inds, device, num_workers, batch_size):
        print("inside hess where N is ", len(subset_inds))
        dummy_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, num_workers=num_workers,
                                                   sampler=SubsetRandomSampler(subset_inds))

        idx = 0
        for in_tensor, target in tqdm(dummy_loader):

            in_tensor, target = in_tensor.to(device), target.to(device)
            output = self._model(in_tensor)
            if idx == 0:
                loss = torch.nn.functional.cross_entropy(output, target, reduction='sum')
            else:
                loss += torch.nn.functional.cross_entropy(output, target, reduction='sum')
            idx += 1

        loss /= len(subset_inds)
        loss.backward(create_graph=True)
        self._compute_second_derivatives()
        self._release_grads()

    def on_epoch_begin(self, dset, subset_inds, device, num_workers, batch_size, epoch_num, **kwargs):

        ####### meta for TrainingProgressTracker ######
        meta = {
            'hess_diag_negatives': {}
        }
        ###############################################

        if self._pruner_not_active(epoch_num):
            return False, {}
        self._add_hess_attr()

        # ensure that the model is not in training mode, this is importance, because
        # otherwise the pruning procedure will interfere and affect the batch-norm statistics
        assert not self._model.training

        self._compute_diag_hessian(dset, subset_inds, device, num_workers, batch_size)
        for idx, module in enumerate(self._modules):
            level = self._required_sparsity(epoch_num)
            w_stat, b_stat = self._get_param_stat(module.weight, module.weight_mask), \
                             self._get_param_stat(module.bias, module.bias_mask)
            module.weight_mask, module.bias_mask = self._get_pruning_mask(w_stat, level), \
                                                   self._get_pruning_mask(None if self._weight_only else b_stat, level)

            ############# adding proportion of negatives in diag hessian meta ############
            total_negatives, total = (module.weight.hess_diag < 0).sum().int(), \
                                     module.weight.numel()
            if module.bias_mask is not None:
                total_negatives += (module.bias.hess_diag < 0).sum().int()
                total += (module.bias.numel())
            meta['hess_diag_negatives'][self._module_names[idx]] = (total_negatives, total)
            ##############################################################################

        self._del_hess_attr()
        return True, meta