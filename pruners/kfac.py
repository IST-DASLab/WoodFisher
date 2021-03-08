import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import time
import logging
from utils.utils import flatten_tensor_list, get_summary_stats, get_total_sparsity
from pruners.woodfisher import WoodburryFisherPruner
import math
from copy import deepcopy

from utils.kfac_utils import (ComputeCovA,
                              ComputeCovG,
                              ComputeCovA_proper,
                              ComputeCovG_proper,
                              StoreA,
                              StoreG,
                              StoreA_proper,
                              StoreG_proper,
                              rm_hooks)
from utils.kfac_fisher_utils import get_blockwise_kfac_inverse, convert_to_kro_dic, get_module_keys, get_module_custom_names
from utils.checkpoints import get_unwrapped_model

class KFACFisherPruner(WoodburryFisherPruner):
    '''
    Pruner based on blockwise (across layers) version of K-FAC to get Hessian estimates
    '''
    def __init__(self, model, inp_args, **kwargs):
        super(KFACFisherPruner, self).__init__(model, inp_args, **kwargs)

        print("IN KFAC")
        self._fisher_inv_diag = None
        self._prune_direction = inp_args.prune_direction
        self._zero_after_prune = inp_args.zero_after_prune
        self._inspect_inv = inp_args.inspect_inv
        self._fisher_mini_bsz = inp_args.fisher_mini_bsz
        if self._fisher_mini_bsz < 0:
            self._fisher_mini_bsz = 1
        if self.args.woodburry_joint_sparsify:
            self._param_stats = []
        if self.args.dump_fisher_inv_mat:
            self._all_grads = []
        self.s_gg, self.s_aa = {}, {}
        self.m_gg, self.m_aa = {}, {}

    ## Input and Output hooks
    def _save_input(self, ActHandler):
        def hook(module, input):
            aa = ActHandler(input[0].data, module)
            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module.custom_name] = torch.diag(aa.new(aa.size(0)).fill_(0))
            self.m_aa[module.custom_name] += aa

        return hook

    def _save_input_offdiagonal(self, ActHandler):
        def hook(module, input):
            a = ActHandler(input[0].data, module)
            print(self.steps, a.shape, 'act a shape')
            # Initialize buffers
            if self.steps == 0:
                self.s_aa[module.custom_name] = a
            else:
                self.s_aa[module.custom_name] = torch.cat([self.s_aa[module.custom_name], a], 0)

        return hook

    def _save_grad_output(self, GradHandler, batch_averaged=True):
        def hook(module, grad_input, grad_output):
            # Accumulate statistics for Fisher matrices
            gg = GradHandler(grad_output[0].data, module, batch_averaged)
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module.custom_name] = torch.diag(gg.new(gg.size(0)).fill_(0))
            self.m_gg[module.custom_name] += gg

        return hook

    def _save_grad_output_offdiagonal(self, GradHandler, batch_averaged=True):
        def hook(module, grad_input, grad_output):
            # Accumulate statistics for Fisher matrices
            g = GradHandler(grad_output[0].data, module, batch_averaged)
            print(self.steps, g.shape, 'grad g shape')
            # Initialize buffers
            if self.steps == 0:
                self.s_gg[module.custom_name] = g
            else:
                self.s_gg[module.custom_name] = torch.cat([self.s_gg[module.custom_name], g], 0)

        return hook
    def _prepare_model(self, offdiagonal=False, proper=True, fix_layers=0):
        count = 0
        print(self._model)
        modules = []
        module_names = []
        print("=> We keep following layers in self._model. ")
        known_modules = {'WrappedLayer'}

        if proper:
            self.CovAHandler = ComputeCovA_proper()
            self.CovGHandler = ComputeCovG_proper()
            self.StoreAHandler = StoreA_proper()
            self.StoreGHandler = StoreG_proper()
        else:
            self.CovAHandler = ComputeCovA()
            self.CovGHandler = ComputeCovG()
            self.StoreAHandler = StoreA()
            self.StoreGHandler = StoreG()

        if not offdiagonal:
            inp_hooks = [self._save_input(self.CovAHandler)]
            out_hooks = [self._save_grad_output(self.CovGHandler)]
        else:
            inp_hooks = []
            out_hooks = []

        if offdiagonal:
            assert self.s_aa is not None
            assert self.s_gg is not None
            inp_hooks.append(self._save_input_offdiagonal(self.StoreAHandler))
            out_hooks.append(self._save_grad_output_offdiagonal(self.StoreGHandler))

        # just going over the modules that have to be pruned
        for module_name, module in self._model.named_modules():
            # print("this is module_name", module_name)
            if module_name not in self._module_names:
                continue
            classname = module.__class__.__name__
            print(module_name, "classname is ", classname)
            if classname in known_modules:
                # print("this module is known ", module_name)
                modules.append(module)
                module_names.append(module_name)
                module.custom_name = module_name # setting custom name
                for inp_hook in inp_hooks:
                    module.register_forward_pre_hook(inp_hook)
                for out_hook in out_hooks:
                    module.register_backward_hook(out_hook)
                print('(%s): %s' % (count, module))
                count += 1
        modules = modules[fix_layers:]
        module_names = module_names[fix_layers:]
        return modules, module_names

    def _compute_woodburry_fisher_inverse(self, dset, subset_inds, device, num_workers, debug=False):
        st_time = time.perf_counter()

        # ensure that the model is not in training mode, this is importance, because
        # otherwise the pruning procedure will interfere and affect the batch-norm statistics
        assert not self._model.training

        self._model = self._model.to(device)
        # self._model = get_unwrapped_model(self._model)

        print("in kfac fisher: len of subset_inds is ", len(subset_inds))

        self.s_aa, self.s_gg = {}, {}
        self.m_aa, self.m_gg = {}, {}

        offdiagonal = False
        self._prepare_model(offdiagonal=offdiagonal)
        self.steps = 0


        goal = self.args.fisher_subsample_size

        assert len(subset_inds) == goal * self.args.fisher_mini_bsz

        # print("# of examples done {} and the goal is {}".format(num, goal))

        print(f"fisher_mini_bsz is {self._fisher_mini_bsz}")

        dummy_loader = torch.utils.data.DataLoader(dset, batch_size=self._fisher_mini_bsz, num_workers=num_workers,
                                                   sampler=SubsetRandomSampler(subset_inds))
        if self.args.aux_gpu_id != -1:
            aux_device = torch.device('cuda:{}'.format(self.args.aux_gpu_id))
        else:
            aux_device = torch.device('cpu')

        if self.args.disable_log_soft:
            # set to true for resnet20 case
            # set to false for mlpnet as it then returns the log softmax and we go to NLL
            criterion = torch.nn.functional.cross_entropy
        else:
            criterion = F.nll_loss

        self._fisher_inv = None

        num_batches = 0
        num_samples = 0

        for in_tensor, target in dummy_loader:
            self._release_grads()

            in_tensor, target = in_tensor.to(device), target.to(device)
            output = self._model(in_tensor)
            loss = criterion(output, target)

            # The default reduction = 'mean' will be used, over the fisher_mini_bsz,
            # which is just a practical heuristic to utilize more datapoints

            if self.args.true_fisher == 'true':
                sampled_y = torch.multinomial(torch.nn.functional.softmax(output.cpu().data, dim=1),
                                              1).squeeze().to(device)
                loss_sample = criterion(output, sampled_y)
                loss_sample.backward()
            else:
                loss = criterion(output, target)
                loss.backward()
            self.steps += 1

            num_batches += 1
            num_samples += self._fisher_mini_bsz
            # print("# of examples done {} and the goal is {}".format(num, goal))


            if num_samples == goal * self._fisher_mini_bsz:
                break

        print("num_samples are", num_samples)
        print(self.s_aa.keys())
        print(self.m_aa.keys())

        self._num_params = 0
        for i, mod_i in enumerate(get_module_custom_names(self._model, self._module_names)):
            self._num_params += self.m_aa[mod_i].shape[0] * self.m_gg[mod_i].shape[0]

        print(f"num_params is {self._num_params}")
        self.m_aa, self.m_gg = convert_to_kro_dic(self._model, self.m_aa, self._module_names), convert_to_kro_dic(self._model,
                                                                                                  self.m_gg, self._module_names)
        print(self.m_aa.keys())
        self._fisher_inv = get_blockwise_kfac_inverse(self._model, m_aa=self.m_aa, m_gg=self.m_gg,
                         num_samples=num_samples, damp=self.args.fisher_damp, use_pi=self.args.kfac_pi, offload_cpu=self.args.offload_inv)

        print("model", self._model)
        self._model = rm_hooks(self._model)

        print(self._model)
        # assert num_samples == goal * self._fisher_mini_bsz
        print("# of examples done {} and the goal (#outer products) is {}".format(num_samples, goal))
        print("# of batches done {}".format(num_batches))
        self._fisher_inv_diag = self._fisher_inv.diagonal()
        print("shape of fisher_inv_diag is ", self._fisher_inv_diag.shape)
        end_time = time.perf_counter()
        print("Time taken to compute fisher inverse with KFAC is {} seconds".format(str(end_time - st_time)))
