import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import time
import logging
from utils.utils import flatten_tensor_list, get_summary_stats
from policies.pruners import GradualPruner
import math

class BlockwiseWoodburryTaylorPruner(GradualPruner):
    def __init__(self, model, inp_args, **kwargs):
        super(BlockwiseWoodburryTaylorPruner, self).__init__(model, inp_args, **kwargs)
        logging.info("IN BLOCK WOODBURRY TAYLOR")
        self._all_grads = []
        self._all_grads_dic = {}

        self._block_fisher_inv_dic = {}
        self._block_fisher_inv_diag_dic = {}
        self._block_fisher_inv_grad_prod_dic = {}
        self._block_fisher_inv_grad_sq_dic = {}

        self._param_stats = []
        # need to consider the update  - ( w_q  (H^-1 e_q ) / [H^-1]_qq
        self._prune_direction = inp_args.prune_direction
        self._zero_after_prune = inp_args.zero_after_prune
        self._inspect_inv = inp_args.inspect_inv
        if self._inspect_inv:
            self.inspect_dic = {}
        self._fisher_mini_bsz = inp_args.fisher_mini_bsz
        if self._fisher_mini_bsz < 0:
            self._fisher_mini_bsz = 1
        self._total_block_fisher_params = 0

        if hasattr(inp_args, 'fittable_params'):
            self._fittable_params = inp_args.fittable_params
        else:
            self._fittable_params = 64 * 64 * 3

        self._avg_grads = None
        self._avg_grads_dic = {}

    def _save_grads(self, loss):
        ys = loss
        params = []
        for module in self._modules:
            for name, param in module.named_parameters():
                # print("name is {} and shape of param is {} \n".format(name, param.shape))

                if self._weight_only and 'bias' in name:
                    continue
                else:
                    params.append(param)

        grads= torch.autograd.grad(ys, params) # first order gradient

        # Do gradient_masking: mask out the parameters which have been pruned previously
        # to avoid rogue calculations for the hessian

        for idx, module in enumerate(self._modules):
            grads[idx].data.mul_(module.weight_mask)

        grads = flatten_tensor_list(grads)

        if self.args.offload_grads:
            grads = grads.cpu()

        if not hasattr(self, '_all_grads'):
            self._all_grads = []

        if self.args.grad_subsample_size is not None:
            # since already averaged over the fisher_mini_bsz
            if self._avg_grads is None:
                self._avg_grads = (grads)/ (self.args.grad_subsample_size/self.args.fisher_mini_bsz)
            else:
                self._avg_grads += (grads)/ (self.args.grad_subsample_size/self.args.fisher_mini_bsz)

            # keep saving gradients as long as less than that needed for fisher inverse (self._goal)
            # otherwise just obtain a more accurate estimate of the gradient

            if len(self._all_grads) <= self._goal:
                self._all_grads.append(grads)

        else:
            self._all_grads.append(grads)
        if not hasattr(self, '_num_params'):
            self._num_params = len(grads)


    def _get_pruned_wts_scaled_basis(self, pruned_params, flattened_params, param_idx):
        return -1 * torch.div(
                torch.mul(pruned_params, (flattened_params - self._block_fisher_inv_grad_prod_dic[param_idx].to(flattened_params.device))),
                self._block_fisher_inv_diag_dic[param_idx].to(flattened_params.device)
        )

    @staticmethod
    def _get_param_stat(param, param_mask, fisher_inv_diag, fisher_inv_grad_prod, fisher_inv_grad_sq, param_idx, woodtaylor_abs=False):
        if param is None or param_mask is None: return None
        # w_i **2 x ((F)^-1)_ii,
        inv_fisher_diag_entry = fisher_inv_diag.view_as(param).to(param.device)

        term1 = 0.5 * ((param ** 2)/(inv_fisher_diag_entry + 1e-10) + 1e-10)

        ehf = fisher_inv_grad_prod.view_as(param)
        ehf = ehf.to(param.device)

        term2 = 0.5 * torch.div(ehf ** 2, (inv_fisher_diag_entry + 1e-10))
        term3 = - 0.5 * fisher_inv_grad_sq.to(param.device)
        term4 = - torch.div(torch.mul(param, ehf), (inv_fisher_diag_entry + 1e-10))
        logging.info(
        "mean value of statistic without eps = {} is {}".format(1e-10, torch.mean(term1 + term2 + term3 + term4).item()))
        logging.info(
            "std value of statistic without eps = {} is {}".format(1e-10, torch.std(term1 + term2 + term3 + term4).item()))

        # this following part doesn't really change things
        logging.info(f"(mean) values of each of terms are: term1 {term1.mean()}, term2 {term2.mean()}, term3 {term3} and term4 {term4.mean()}")
        logging.info(
            f"(std) values of each of terms are: term1 {term1.std()}, term2 {term2.std()}, term3 0.0 and term4 {term4.std()}")
        stat = term1 + term2 + term3 + term4
        logging.info(f"is woodtaylor pruning statistic positive: {(stat >= 0).all()}")
        logging.info(f"(before min subtract) woodtaylor pruning statistic mean: {torch.mean(stat).item()} and std: {torch.std(stat).item()}")
        if woodtaylor_abs:
            stat = torch.abs(stat)
        stat = stat - stat.min() + 1e-10
        logging.info(
            "mean value of (min subtracted) statistic without eps = {} is {}".format(1e-10, torch.mean(stat).item()))
        logging.info(
            "std value of (min subtracted) statistic without eps = {} is {}".format(1e-10, torch.std(stat).item()))
        return stat * param_mask


    def _release_grads(self):
        optim.SGD(self._model.parameters(), lr=1e-10).zero_grad()

    def _add_outer_products_efficient_v1(self, mat, vec, num_parts=2, scale=1.0):
        piece = int(math.ceil(len(vec) / num_parts))
        vec_len = len(vec)
        for i in range(num_parts):
            for j in range(num_parts):
                mat[i * piece:min((i + 1) * piece, vec_len), j * piece:min((j + 1) * piece, vec_len)].add_(
                    torch.ger(vec[i * piece:min((i + 1) * piece, vec_len)], vec[j * piece:min((j + 1) * piece, vec_len)])/scale
                )

    def _compute_all_grads(self, dset, subset_inds, device, num_workers, debug=False):
        st_time = time.perf_counter()
        self._model = self._model.to(device)

        logging.info(f"computing all grads for blockwise taylorwood: len of subset_inds is {len(subset_inds)}")

        self._goal = self.args.fisher_subsample_size

        assert len(subset_inds) == self._goal * self._fisher_mini_bsz

        # print("# of examples done {} and the self._goal is {}".format(num, self._goal))

        dummy_loader = torch.utils.data.DataLoader(dset, batch_size=self._fisher_mini_bsz, num_workers=num_workers,
                                                   sampler=SubsetRandomSampler(subset_inds))

        if self.args.disable_log_soft:
            # set to true for resnet20 case
            # set to false for mlpnet as it then returns the log softmax and we go to NLL
            criterion = torch.nn.functional.cross_entropy
        else:
            criterion = F.nll_loss

        num_batches = 0
        num_samples = 0

        for in_tensor, target in dummy_loader:
            self._release_grads()

            in_tensor, target = in_tensor.to(device), target.to(device)
            output = self._model(in_tensor)

            if self.args.true_fisher:
                sampled_y = torch.multinomial(torch.nn.functional.softmax(output.cpu().data, dim=1), 1).squeeze(1).to(
                    device)
                loss = criterion(output, sampled_y)
            else:
                loss = criterion(output, target)

            # The default reduction = 'mean' will be used, over the fisher_mini_bsz,
            # which is just a practical heuristic to utilize more datapoints

            self._save_grads(loss)

            num_batches += 1
            num_samples += self._fisher_mini_bsz
            # print("# of examples done {} and the self._goal is {}".format(num, self._goal))

            if self.args.grad_subsample_size is None:
                if num_samples == (self._goal * self._fisher_mini_bsz):
                    break
            else:
                if num_samples == self.args.grad_subsample_size:
                    break

        if self.args.grad_subsample_size is None:
            assert num_samples == (self._goal * self._fisher_mini_bsz)
        else:
            assert num_samples == self.args.grad_subsample_size
        logging.info('Check what is grad_subsample_size: {}'.format(self.args.grad_subsample_size))
        logging.info("# of examples done {} and the self._goal is {}".format(num_samples, self._goal))
        logging.info("# of batches done {}".format(num_batches))

        end_time = time.perf_counter()
        logging.info("Time taken to save all grads is {} seconds".format(str(end_time - st_time)))


    def _organize_grads(self):
        self._all_grads = torch.stack(self._all_grads)

        _param_count = 0
        for idx, module in enumerate(self._modules):
            self._all_grads_dic[_param_count] = self._all_grads[:, _param_count:_param_count + module.weight.numel()]

            if self.args.grad_subsample_size is not None:
                self._avg_grads_dic[_param_count] = self._avg_grads[_param_count:_param_count + module.weight.numel()]

            _param_count += module.weight.numel()

        assert _param_count == self._num_params
        self._num_grads = self._all_grads.shape[0]
        del self._all_grads
        del self._avg_grads
        torch.cuda.empty_cache()

    def _compute_split_block_woodburry_taylor_inverse(self, param_idx, start, end, device):

        st_time = time.perf_counter()

        _sub_block_fisher_inv = None
        if self.args.grad_subsample_size is None:
            _sub_block_avg_grads = None
        else:
            _sub_block_avg_grads = self._avg_grads_dic[param_idx][start:end]

        if self.args.aux_gpu_id != -1:
            aux_device = torch.device('cuda:{}'.format(self.args.aux_gpu_id))
        else:
            aux_device = torch.device('cpu')
        logging.info(f'In sub_block for param_idx {param_idx}, with split indices: start {start} and end {end}')
        logging.info(f"{self._num_grads}, len (number) of grads")

        for idx in range(self._num_grads):

            sample_grads = self._all_grads_dic[param_idx][idx][start:end]

            if self.args.fisher_cpu:
                sample_grads = sample_grads.cpu()

            if aux_device is not None and aux_device != torch.device('cpu'):
                sample_grads = sample_grads.to(aux_device)

            if not self.args.fisher_cpu and sample_grads.device == torch.device('cpu'):
                sample_grads = sample_grads.to(device)

            if self.args.grad_subsample_size is None:
                if _sub_block_avg_grads is None:
                    _sub_block_avg_grads = sample_grads
                else:
                    _sub_block_avg_grads += sample_grads

            # print(f'device of sample_grads is {sample_grads.device}')
            if idx == 0:
                # rewrite in terms of inplace operations

                numerator_normalization = (self.args.fisher_damp) ** 2

                _sub_block_fisher_inv = torch.ger(sample_grads, sample_grads).mul_(1.0 / numerator_normalization).div_(
                    self._goal + (sample_grads.dot(sample_grads) / self.args.fisher_damp)
                )
                _sub_block_fisher_inv.diagonal().sub_(1.0 / self.args.fisher_damp)
                # 1/self.args.fisher_damp \times Identity matrix is used to represent (H^-1)_0
                _sub_block_fisher_inv.mul_(-1)

            else:
                cache_matmul = torch.matmul(_sub_block_fisher_inv, sample_grads)
                cache_matmul.div_((self._goal + sample_grads.dot(cache_matmul)) ** 0.5)

                if not self.args.fisher_optimized:
                    _sub_block_fisher_inv.sub_(
                        torch.ger(cache_matmul, cache_matmul)
                        # / (goal + sample_grads.dot(cache_matmul))
                    )
                else:
                    assert self.args.fisher_parts > 1
                    # F = F - x x'
                    # F1 = -F
                    _sub_block_fisher_inv.mul_(-1)
                    # F1 + x x'
                    self._add_outer_products_efficient_v1(
                        _sub_block_fisher_inv, cache_matmul, num_parts=self.args.fisher_parts
                    )
                    # F = - F1
                    _sub_block_fisher_inv.mul_(-1)

                del cache_matmul

                del sample_grads

        if self.args.grad_subsample_size is None:
            _sub_block_avg_grads /= self._num_grads
        end_time = time.perf_counter()
        logging.info("Time taken to compute the sub block_fisher_inverse for param idx {} with woodburry is {} seconds".format(
            param_idx, str(end_time - st_time)))

        if self.args.centered:
            # subtract the full gradient outer product
            # because, we subtract the signs in woodbury update would change
            # hence I have add_ instead of sub_ (in the non fisher_optimized case)
            # and I remove the two mul_(-1) and sub_() for the other case

            cache_matmul = torch.matmul(_sub_block_fisher_inv, _sub_block_avg_grads.to(_sub_block_fisher_inv.device))
            # minus because we subtract the full gradient outer product
            cache_scalar = (1 - (_sub_block_avg_grads.to(cache_matmul.device)).dot(cache_matmul))

            if not self.args.fisher_optimized:
                _sub_block_fisher_inv.add_(
                    torch.ger(cache_matmul, cache_matmul) / cache_scalar
                )
            else:
                assert self.args.fisher_parts > 1
                self._add_outer_products_efficient_v1(
                    _sub_block_fisher_inv, cache_matmul, num_parts=self.args.fisher_parts, scale=cache_scalar
                )

            del cache_matmul
            del cache_scalar

        if self.args.offload_inv:
            _sub_block_fisher_inv = _sub_block_fisher_inv.cpu()

        _sub_block_fisher_inv_grad_prod = _sub_block_fisher_inv @ _sub_block_avg_grads.to(_sub_block_fisher_inv.device)
        _sub_block_fisher_inv_grad_sq = torch.dot(_sub_block_fisher_inv_grad_prod,
                                                  _sub_block_avg_grads.to(_sub_block_fisher_inv.device))

        return _sub_block_fisher_inv, _sub_block_fisher_inv_grad_prod, _sub_block_fisher_inv_grad_sq

    def _compute_block_woodburry_taylor_inverse(self, param, param_idx, device):

        st_time = time.perf_counter()

        _block_fisher_inv = None
        if self.args.grad_subsample_size is None:
            _block_avg_grads = None
        else:
            _block_avg_grads = self._avg_grads_dic[param_idx]

        if self.args.aux_gpu_id != -1:
            aux_device = torch.device('cuda:{}'.format(self.args.aux_gpu_id))
        else:
            aux_device = torch.device('cpu')

        logging.info(f"{self._num_grads} len (number) of grads")

        if not self.args.fisher_split_grads or param.numel() <= self._fittable_params:
            for idx in range(self._num_grads):
                sample_grads = self._all_grads_dic[param_idx][idx]

                if self.args.grad_subsample_size is None:
                    if _block_avg_grads is None:
                        _block_avg_grads = sample_grads
                    else:
                        _block_avg_grads += sample_grads

                if self.args.fisher_cpu:
                    sample_grads = sample_grads.cpu()

                if aux_device is not None and aux_device != torch.device('cpu'):
                    sample_grads = sample_grads.to(aux_device)

                if not self.args.fisher_cpu and sample_grads.device == torch.device('cpu'):
                    sample_grads = sample_grads.to(device)

                # print(f'device of sample_grads is {sample_grads.device}')
                if idx == 0:
                    # rewrite in terms of inplace operations
                    numerator_normalization = (self.args.fisher_damp) ** 2

                    _block_fisher_inv = torch.ger(sample_grads, sample_grads).mul_(1.0 / numerator_normalization).div_(
                        self._goal + (sample_grads.dot(sample_grads) / self.args.fisher_damp)
                    )
                    _block_fisher_inv.diagonal().sub_(1.0 / self.args.fisher_damp)
                    # 1/self.args.fisher_damp \times Identity matrix is used to represent (H^-1)_0
                    _block_fisher_inv.mul_(-1)

                else:
                    cache_matmul = torch.matmul(_block_fisher_inv, sample_grads)
                    cache_matmul.div_((self._goal + sample_grads.dot(cache_matmul)) ** 0.5)

                    if not self.args.fisher_optimized:
                        _block_fisher_inv.sub_(
                            torch.ger(cache_matmul, cache_matmul)
                        )
                    else:
                        assert self.args.fisher_parts > 1
                        # F = F - x x'
                        # F1 = -F
                        _block_fisher_inv.mul_(-1)
                        # F1 + x x'
                        self._add_outer_products_efficient_v1(
                            _block_fisher_inv, cache_matmul, num_parts=self.args.fisher_parts
                        )
                        #F = - F1
                        _block_fisher_inv.mul_(-1)

                    del cache_matmul

                    del sample_grads

            if self.args.grad_subsample_size is None:
                _block_avg_grads /= self._num_grads
            logging.info(f'norm of avg_grads is {torch.norm(_block_avg_grads).item()}')

            if self.args.centered:
                # subtract the full gradient outer product
                # because, we subtract the signs in woodbury update would change
                # hence I have add_ instead of sub_ (in the non fisher_optimized case)
                # and I remove the two mul_(-1) and sub_() for the other case

                cache_matmul = torch.matmul(_block_fisher_inv, _block_avg_grads.to(_block_fisher_inv.device))
                # minus because we subtract the full gradient outer product
                cache_scalar = (1 - (_block_avg_grads.to(cache_matmul.device)).dot(cache_matmul))

                if not self.args.fisher_optimized:
                    _block_fisher_inv.add_(
                        torch.ger(cache_matmul, cache_matmul)/cache_scalar
                        # / (goal + sample_grads.dot(cache_matmul))
                    )
                else:
                    assert self.args.fisher_parts > 1
                    self._add_outer_products_efficient_v1(
                        _block_fisher_inv, cache_matmul, num_parts=self.args.fisher_parts, scale=cache_scalar
                    )

                del cache_matmul
                del cache_scalar

            # param_idx is the index of the first parameter of this layer
            # in other words, count of parameters before this!
            if self.args.offload_inv:
                _block_fisher_inv = _block_fisher_inv.cpu()

            self._block_fisher_inv_diag_dic[param_idx] = _block_fisher_inv.diagonal()
            self._block_fisher_inv_dic[param_idx] = _block_fisher_inv

            self._block_fisher_inv_grad_prod_dic[param_idx] = _block_fisher_inv @ _block_avg_grads.to(_block_fisher_inv.device)
            self._block_fisher_inv_grad_sq_dic[param_idx] = torch.dot(self._block_fisher_inv_grad_prod_dic[param_idx], _block_avg_grads.to(_block_fisher_inv.device))

        else:
            # if the number of params is > fittable parameter limit
            num_params = param.numel()
            num_splits = int(math.ceil(num_params/self._fittable_params))
            _block_fisher_inv = []
            _block_fisher_inv_diag = []
            _block_fisher_inv_grad_prod = []
            _block_fisher_inv_grad_sq = []
            split_start = 0
            for split_idx in range(num_splits):
                split_end = min((split_idx + 1) * self._fittable_params, num_params)
                _sub_block_fisher_inv, _sub_block_fisher_inv_grad_prod, _sub_block_fisher_inv_grad_sq = self._compute_split_block_woodburry_taylor_inverse(param_idx, split_start, split_end, device)

                _block_fisher_inv.append(_sub_block_fisher_inv)
                _block_fisher_inv_diag.append(_block_fisher_inv[-1].diagonal())

                _block_fisher_inv_grad_prod.append(_sub_block_fisher_inv_grad_prod)
                _block_fisher_inv_grad_sq.append(_sub_block_fisher_inv_grad_sq)

                split_start += min(self._fittable_params, num_params-split_start)
            assert split_start == num_params

            _block_fisher_inv_diag = torch.cat(_block_fisher_inv_diag)
            _block_fisher_inv_grad_prod = torch.cat(_block_fisher_inv_grad_prod)
            _block_fisher_inv_grad_sq = torch.Tensor(_block_fisher_inv_grad_sq).sum().float().to(_block_fisher_inv_diag.device)

            self._block_fisher_inv_dic[param_idx] = _block_fisher_inv
            self._block_fisher_inv_diag_dic[param_idx] = _block_fisher_inv_diag
            self._block_fisher_inv_grad_prod_dic[param_idx] = _block_fisher_inv_grad_prod
            self._block_fisher_inv_grad_sq_dic[param_idx] = _block_fisher_inv_grad_sq

        self._total_block_fisher_params += len(self._block_fisher_inv_diag_dic[param_idx])

        end_time = time.perf_counter()
        logging.info("Time taken to compute one block_fisher_inverse for param idx {} with woodburry is {} seconds".format(param_idx, str(end_time - st_time)))

        if self._inspect_inv:
            logging.info(f'---- Inspecting block fisher inverse for param_idx {idx} ----')
            inspect_dic = get_summary_stats(self._block_fisher_inv)
            inspect_dic['trace'] = self._block_fisher_inv.trace().item()
            inspect_dic['sum'] = self._block_fisher_inv.sum().item()
            inspect_dic['trace/sum'] = (self._block_fisher_inv.trace()/self._block_fisher_inv.sum()).item()
            self.inspect_dic[param_idx] = inspect_dic
            logging.info(f"{self.inspect_dic[param_idx]}")
            logging.info('-----------------------------------')

    def _compute_single_layer_trace(self, grads):
        logging.info("shape of grads is {}".format(grads.shape))
        return (torch.norm(grads, dim=1)**2).sum()/grads.shape[0]

    def _compute_layerwise_fisher_traces(self):
        self._trace_dic = {}
        _param_count = 0
        for idx, module in enumerate(self._modules):
            self._trace_dic[self._module_names[idx]] = self._compute_single_layer_trace(self._all_grads_dic[_param_count])
            _param_count += module.weight.numel()
        assert _param_count == self._num_params
        logging.info("This is the dictionary containing layerwise traces {}".format(self._trace_dic))

    def _get_weight_update(self, param_idx, scaled_basis_vector, param_norm):
        scaled_basis_device = scaled_basis_vector.device

        if not self.args.fisher_split_grads or type(self._block_fisher_inv_dic[param_idx])!=list:
            weight_update = self._block_fisher_inv_dic[param_idx] @ scaled_basis_vector.to(self._block_fisher_inv_dic[param_idx].device)
        else:
            weight_update = []
            split_start = 0
            # print(f"shape of scaled_basis vector at param_idx {param_idx} is", scaled_basis_vector.shape)
            for split_idx in range(len(self._block_fisher_inv_dic[param_idx])):
                # print(f"shape of subblock at split_idx {split_idx} is ", self._block_fisher_inv_dic[param_idx][split_idx].shape)
                split_len = self._block_fisher_inv_dic[param_idx][split_idx].shape[0]
                weight_update.append(self._block_fisher_inv_dic[param_idx][split_idx] @
                                     scaled_basis_vector[split_start:split_start + split_len].to(self._block_fisher_inv_dic[param_idx][split_idx].device))
                split_start += split_len

            assert split_start == scaled_basis_vector.shape[0]
            weight_update = torch.cat(weight_update)

        weight_update = weight_update.to(scaled_basis_device)

        # woodtaylor: subtract the hess inverse times grad term
        logging.info(f'for param {param_idx}: norm of weight update before is {torch.norm(weight_update).item()}')

        weight_update -= self._block_fisher_inv_grad_prod_dic[param_idx].to(scaled_basis_vector.device)
        logging.info(f'for param {param_idx}: norm of weight update (after subtracting hessinv grad term) is {torch.norm(weight_update).item()}')

        if self.args.normalize_update:
            if torch.norm(weight_update) > param_norm:
                weight_update /= (torch.norm(weight_update) / (param_norm * self.args.normalize_update_mult)  + 1e-9)
                logging.info(f'for param {param_idx}: norm of weight update normalizing it is {torch.norm(weight_update).item()}')
            else:
                logging.info('it is fine, update has smaller norm than that of the weights')

        return weight_update

    def on_epoch_begin(self, dset, subset_inds, device, num_workers, epoch_num, **kwargs):
        meta = {}
        if self._pruner_not_active(epoch_num):
            logging.info("Pruner is not ACTIVEEEE yaa!")
            return False, {}

        # ensure that the model is not in training mode, this is importance, because
        # otherwise the pruning procedure will interfere and affect the batch-norm statistics
        assert not self._model.training

        # reinit params if were deleted during gradual pruning
        if not hasattr(self, '_all_grads'):
            self._all_grads = []
        if not hasattr(self, '_all_grads_dic'):
            self._all_grads_dic = {}
        if not hasattr(self, '_block_fisher_inv_dic'):
            self._block_fisher_inv_dic = {}
        if not hasattr(self, '_block_fisher_inv_diag_dic'):
            self._block_fisher_inv_diag_dic = {}
        if not hasattr(self, '_block_fisher_inv_grad_prod_dic'):
            self._block_fisher_inv_grad_prod_dic = {}
        if not hasattr(self, '_block_fisher_inv_grad_sq_dic'):
            self._block_fisher_inv_grad_sq_dic = {}
        if not hasattr(self, '_param_stats'):
            self._param_stats = []
        if not hasattr(self, '_avg_grads'):
            self._avg_grads = None
        if not hasattr(self, '_avg_grads_dic'):
            self._avg_grads_dic = {}

        self._total_block_fisher_params = 0

        # Step0: compute all grads!
        self._compute_all_grads(dset, subset_inds, device, num_workers)

        # Step0.5: organize all grads into a dic!
        self._organize_grads()
        self._release_grads()

        if self.args.fisher_trace:
            self._compute_layerwise_fisher_traces()

        if self._prune_direction:
            meta['prune_direction'] = []
            meta['original_param'] = []
            meta['mask_previous'] = []
            meta['mask_overall'] = []

        self._param_idx = 0

        woodtaylor_abs = self.args.woodtaylor_abs
        # Step1: compute blockwise wood fisher inverse!
        for idx, module in enumerate(self._modules):
            # print(f'module is {module}')
            self._compute_block_woodburry_taylor_inverse(module.weight, self._param_idx, device)
            self._all_grads_dic.pop(self._param_idx, None)
            # del self._all_grads_dic[self._param_idx]
            torch.cuda.empty_cache()

            if self.args.fisher_cpu:
                w_stat = self._get_param_stat(module.weight.cpu(), module.weight_mask.cpu(),
                                              self._block_fisher_inv_diag_dic[self._param_idx], self._block_fisher_inv_grad_prod_dic[self._param_idx],
                                              self._block_fisher_inv_grad_sq_dic[self._param_idx], self._param_idx, woodtaylor_abs=woodtaylor_abs)
            else:
                w_stat = self._get_param_stat(module.weight, module.weight_mask, self._block_fisher_inv_diag_dic[self._param_idx], self._block_fisher_inv_grad_prod_dic[self._param_idx],
                                              self._block_fisher_inv_grad_sq_dic[self._param_idx], self._param_idx, woodtaylor_abs=woodtaylor_abs)

            if not self.args.woodburry_joint_sparsify:
                assert self._weight_only
                weight_mask_before = module.weight_mask
                level = self._required_sparsity(epoch_num)
                module.weight_mask, module.bias_mask = self._get_pruning_mask(w_stat, level), None

                if self.args.fisher_cpu:
                    weight_mask_before = weight_mask_before.cpu()

                pruned_weights = weight_mask_before - module.weight_mask
                prune_mask = weight_mask_before > module.weight_mask  # active if was 1 before and 0 now
                pruned_weights = pruned_weights.flatten().float()

                # print(f'weights in parameter {idx} named {self._module_names[idx]} before pruning (only for pruned) are ', module.weight[prune_mask])

                if not self.args.fisher_cpu:
                    scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights, module.weight.flatten(),
                                                                            self._param_idx)
                else:
                    scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights.cpu(), module.weight.flatten().cpu(),
                                                                            self._param_idx)

                # weight_update = self._block_fisher_inv_dic[self._param_idx].to(scaled_basis_vector.device) @ scaled_basis_vector
                weight_update = self._get_weight_update(self._param_idx, scaled_basis_vector, torch.norm(module.weight))
                weight_update = weight_update.view_as(module.weight).to(module.weight.device)
                logging.info(f'at idx {idx} named {self._module_names[idx]}, shape of weight update is {list(weight_update.shape)}')
                if self._zero_after_prune:
                    # This flag is used in case when analyze the loss approximation due to pruning.

                    # It's purpose is to make those active in the prune_mask to be 0 weight
                    # since later module.weight will be added to the weight_update.
                    # because of the effect of separate OBS parameter readjustments terms in practice,
                    # weight update by itself won't result in weight 0 - at the places pruned.

                    # However, for most of the usage, this does not matter, as we multiply weight matrices
                    # by the mask when considering pruning or retraining anyways!

                    weight_update[prune_mask] = (-1 * module.weight.data[prune_mask])

                if self.args.fisher_cpu:
                    weight_update = weight_update.to(module.weight.device)
                    module.weight_mask = module.weight_mask.to(module.weight.device)
                    if module.bias_mask is not None:
                        module.bias_mask = module.bias_mask.to(module.weight.device)

                    print('device of weight_update is ', weight_update.device)
                    print('device of module.weight_mask is ', module.weight_mask.device)

                logging.info(f'for param {idx} named {self._module_names[idx]} before update: norm of weight is {torch.norm(module.weight).item()}')

                with torch.no_grad():
                    module.weight[:] = module.weight.data + weight_update

                logging.info(f'for param {idx} named {self._module_names[idx]}: norm of weight update is {torch.norm(weight_update).item()}')

                if self._prune_direction:
                    meta['prune_direction'].append(weight_update)
                    meta['original_param'].append(module.weight.data.clone())
                    meta['mask_previous'].append(weight_mask_before)
                    meta['mask_overall'].append(module.weight_mask)

                # save memory when doing joint sparsification!
                del w_stat
                del self._block_fisher_inv_diag_dic[self._param_idx]
                del self._block_fisher_inv_dic[self._param_idx]
                del self._block_fisher_inv_grad_prod_dic[self._param_idx]
                del self._block_fisher_inv_grad_sq_dic[self._param_idx]

                del pruned_weights
                del prune_mask
                del scaled_basis_vector
                del weight_update
                del weight_mask_before
            else:
                # w_stat has the shape of module.weight, flatten it
                # later on after obtaining the mask unflatten!
                if self.args.fisher_cpu:
                    w_stat = w_stat.cpu()

                self._param_stats.append(w_stat.flatten())

            self._param_idx += module.weight.numel()

            if module.bias is not None and not self._weight_only:
                logging.info('sparsifying bias as well')
                self._compute_block_woodburry_taylor_inverse(module.bias, self._param_idx, device)
                b_stat = self._get_param_stat(module.bias, module.bias_mask,
                                              self._block_fisher_inv_diag_dic[self._param_idx].to(module.bias.device),
                                              self._block_fisher_inv_grad_prod_dic[self._param_idx],
                                              self._block_fisher_inv_grad_sq_dic[self._param_idx], self._param_idx, woodtaylor_abs=woodtaylor_abs)


                self._param_stats.append(b_stat.flatten())
                self._param_idx += module.bias.numel()



        if self.args.woodburry_joint_sparsify:

            # Step2: doing global parameter selection!
            level = self._required_sparsity(epoch_num)
            global_param_mask = self._get_pruning_mask(flatten_tensor_list(self._param_stats), level)
            logging.info('shape of global param mask is {}'.format(global_param_mask.shape))
            del self._param_stats

            # Step3: computing global update!

            assert self._weight_only
            _param_count = 0
            for idx, module in enumerate(self._modules):
                weight_mask_before = module.weight_mask
                module.weight_mask = global_param_mask[_param_count:_param_count + module.weight.numel()].view_as(module.weight)

                if self.args.fisher_cpu:
                    weight_mask_before = weight_mask_before.cpu()

                pruned_weights = weight_mask_before - module.weight_mask
                prune_mask = weight_mask_before > module.weight_mask # active if was 1 before and 0 now
                pruned_weights = pruned_weights.flatten().float()

                # print(f'weights in parameter {idx} named {self._module_names[idx]} before pruning (only for pruned) are ', module.weight[prune_mask])
                if not self.args.fisher_cpu:
                    scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights, module.weight.flatten(),
                                                                            _param_count)
                else:
                    scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights.cpu(), module.weight.flatten().cpu(),
                                                                            _param_count)

                # weight_update = self._block_fisher_inv_dic[_param_count].to(scaled_basis_vector.device) @ scaled_basis_vector
                weight_update = self._get_weight_update(_param_count, scaled_basis_vector, torch.norm(module.weight))
                weight_update = weight_update.view_as(module.weight)
                logging.info(f'at idx {idx} named {self._module_names[idx]}, shape of weight update is {list(weight_update.shape)}')
                if self._zero_after_prune:
                    # This flag is used in case when analyze the loss approximation due to pruning.

                    # It's purpose is to make those active in the prune_mask to be 0 weight
                    # since later module.weight will be added to the weight_update.
                    # because of the effect of separate OBS parameter readjustments terms in practice,
                    # weight update by itself won't result in weight 0 - at the places pruned.

                    # However, for most of the usage, this does not matter, as we multiply weight matrices
                    # by the mask when considering pruning or retraining anyways!

                    weight_update[prune_mask] = (-1 * module.weight.data[prune_mask])

                if self.args.fisher_cpu:
                    weight_update = weight_update.to(module.weight.device)
                    module.weight_mask = module.weight_mask.to(module.weight.device)
                    if module.bias_mask is not None:
                        module.bias_mask = module.bias_mask.to(module.weight.device)

                    print('device of weight_update is ', weight_update.device)
                    print('device of module.weight_mask is ', module.weight_mask.device)

                logging.info(f'for param {idx} named {self._module_names[idx]} before update: norm of weight is {torch.norm(module.weight).item()}')

                with torch.no_grad():
                    module.weight[:] = module.weight.data + weight_update

                logging.info(f'for param {idx} named {self._module_names[idx]}: norm of weight update is {torch.norm(weight_update).item()}')

                if self._prune_direction:
                    meta['prune_direction'].append(weight_update)
                    meta['original_param'].append(module.weight.data.clone())
                    meta['mask_previous'].append(weight_mask_before)
                    meta['mask_overall'].append(module.weight_mask)

                # print('weight before is ', module.weight)

                # print('weight after is ', module.weight)
                logging.info(f'for param {idx} named {self._module_names[idx]} after update: norm of weight is {torch.norm(module.weight).item()}')

                # print(f'weights in parameter {idx} named {self._module_names[idx]} after pruning (only for pruned) are ', module.weight[prune_mask])

                # inspecting if underlying references did not change
                if self._prune_direction:
                    print(f'weights in meta[original_param][{idx}] after pruning (only for pruned) are ', meta['original_param'][idx])

                self._block_fisher_inv_dic[_param_count] = None
                self._block_fisher_inv_diag_dic[_param_count] = None
                self._block_fisher_inv_grad_prod_dic[_param_count] = None
                self._block_fisher_inv_grad_sq_dic[_param_count] = None

                del pruned_weights
                del prune_mask
                del scaled_basis_vector
                del weight_update
                del weight_mask_before

                _param_count += module.weight.numel()

                module.bias_mask = None
                # PR: later maybe handle bias, for now set bias_mask to None
                # if module.bias is not None and not self._weight_only:
                #     module.bias_mask = global_param_mask[_param_count + module.bias.numel()].view_as(module.bias)
                #     _param_count += module.bias.numel()


            assert self._param_idx == _param_count


        # check if all the params whose fisher inverse was computed their value has been taken
        # print(f'param_idx is {self._param_idx} and fisher_inv_shape[0] is {len(self._all_grads[0])} \n')
        assert self._param_idx == self._total_block_fisher_params
        assert self._num_params == self._total_block_fisher_params

        del self._block_fisher_inv_diag_dic
        del self._block_fisher_inv_dic
        del self._block_fisher_inv_grad_prod_dic
        del self._block_fisher_inv_grad_sq_dic

        if self._inspect_inv:
            meta['inspect_dic'] = self.inspect_dic

        return True, meta