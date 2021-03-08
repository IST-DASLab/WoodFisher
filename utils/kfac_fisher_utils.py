from utils.kfac_utils import (ComputeCovA,
                              ComputeCovG,
                              ComputeCovA_proper,
                              ComputeCovG_proper,
                              StoreA,
                              StoreG,
                              StoreA_proper,
                              StoreG_proper,
                              rm_hooks)

import torch
import torch.optim as optim
import os
import json
from tqdm import tqdm
import copy
import numpy as np

def get_timestamp_other():
    import time
    import datetime
    ts = time.time()
    # %f allows granularity at the micro second level!
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S_%f')
    return timestamp

def mkdir(path):
    os.makedirs(path, exist_ok=True)

class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def tensor_to_list(tensor):
    if len(tensor.shape) == 1:
        return [tensor[_].item() for _ in range(tensor.shape[0])]
    else:
        return [tensor_to_list(tensor[_]) for _ in range(tensor.shape[0])]

def save_tensor(tens, dump_dir, name):
    torch.save(tens, os.path.join(dump_dir, f'{name}.pt'))

def get_num_params(model):
    return sum([module.numel() for module in model.parameters()])

def dump_parameters(args):
    print("dumping parameters at ", args.dump_dir)
    with open(os.path.join(args.dump_dir, 'config.txt'), 'w') as outfile:
        if not (type(args) is dict or type(args) is dotdict):
            json.dump(vars(args), outfile, sort_keys=True, indent=4)
        else:
            json.dump(args, outfile, sort_keys=True, indent=4)

def save_final_model(args, model, optimizer, test_accuracies, dump_name='./kfac_dump', ckpt_type='final'):
    import time
    args.ckpt_type = ckpt_type
    time.sleep(1)  # workaround for RuntimeError('Unknown Error -1') https://github.com/pytorch/pytorch/issues/10577
    curr_timestamp = get_timestamp_other()
    if not hasattr(args, 'dump_dir') or args.dump_dir is None:
        args.dump_dir = os.path.join(dump_name, curr_timestamp)
    mkdir(args.dump_dir)
    dump_parameters(args)
    torch.save({
        'args': vars(args),
        'epoch': args.epochs,
        'test_accuracies': test_accuracies,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, os.path.join(args.dump_dir, 'final_model.pt')
    )
    print("Dumped model and optimizer and other meta info")
    print("The path where it is saved is ", os.path.join(args.dump_dir, 'final_model.pt'))
    print("The args for this experiment were ", args)

def get_pretrained_model_optimizer(args, model, optimizer):

    assert args.load_model != ''
    if os.path.isfile(args.load_model):
        load_path = args.load_model
    elif os.path.isdir(args.load_model):
        load_path = os.path.join(args.load_model, 'final_model.pt')
    if args.gpu_id != -1:
        state = torch.load(
            load_path,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cuda:' + str(args.gpu_id))
            ),
        )
    else:
        state = torch.load(
            load_path,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )

    print("Loading model at path {} which had test accuracy {} at epoch {}".format(load_path, state['test_accuracies'][-1],
                                                                                  state['epoch']))
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])

    if args.gpu_id != -1:
        model = model.cuda(args.gpu_id)

def get_sample_args():
    args = {'enable_dropout': False,
             'MNIST_PATH': '../hessian/files/',
             'to_download': True,
             'batch_size_train': 64,
             'batch_size_test': 1000,
             'train_bsz': 64,
             'test_bsz': 1000,
             'subsample_size': 5000,
             'momentum': 0.5,
             'lr': 0.001,
             'gpu_id': 0,
             'log_interval': 20,
             'dump_model': False,
             'epochs': 10,
             'load_model': './kfac_dump/2020-03-24_18-56-29_616127',
            'num_hidden_nodes1': 40,
            'num_hidden_nodes2': 20,
            'num_classes':10,
            'input_size': 784,
            }

    return dotdict(args)

## Kronecker utils
def kronecker(A, B):
    print(f"shape of A and B is {A.shape} and {B.shape} resp.")
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

def kronecker_shape(shape_a, shape_b, idx=-1):
    if idx == -1:
        assert shape_a[0]*shape_b[0] == shape_a[1]*shape_b[1]
        return shape_a[0]*shape_b[0]
    else:
        return shape_a[idx]*shape_b[idx]


def batch_kronecker(A, B, reduce=None, minibsz=-1):
    # Reduce refers to taking the mean of the batch of kronecker products computed
    # Also, support mini-batching in reduction mode to save memory!
    assert A.size(0) == B.size(0)
    bsz = A.size(0)
    if reduce is None:
        return torch.einsum('bpq, brs->bprqs', A, B).view(bsz, A.size(1) * B.size(1), A.size(2) * B.size(2))
    else:
        if minibsz == -1:
            if reduce == 'mean':
                return torch.einsum('bpq, brs->bprqs', A, B).mean(dim=0).view(A.size(1) * B.size(1),
                                                                              A.size(2) * B.size(2))
            elif reduce == 'sum':
                return torch.einsum('bpq, brs->bprqs', A, B).sum(dim=0).view(A.size(1) * B.size(1),
                                                                             A.size(2) * B.size(2))
        else:
            assert bsz % minibsz == 0
            ans = A.new(A.size(1) * B.size(1), A.size(2) * B.size(2)).fill_(0)
            num_iters = int(bsz / minibsz)
            for idx in range(num_iters):
                ans += torch.einsum('bpq, brs->bprqs',
                                    A[idx * bsz:(idx + 1) * bsz], B[idx * bsz:(idx + 1) * bsz]
                                    ).sum(dim=0).view(A.size(1) * B.size(1), A.size(2) * B.size(2))
            if reduce == 'mean':
                ans /= bsz
            return ans

def batch_outer_product(A, B):
    assert A.size(0) == B.size(0)
    # actually this is more of batch of outer products that I have implemented here in!
    return torch.einsum('bp, bq->bpq', A, B).view(A.size(0), A.size(1)*B.size(1))

## Input and Output hooks
def _save_input(m_aa, ActHandler):
    def hook(module, input):

        aa = ActHandler(input[0].data, module)
        # Initialize buffers
        if steps == 0:
            # basically initializes a matrix of the size of aa,
            # why not simply, xx.new(xx.size()).fill_(0)?
            m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(0))
        m_aa[module] += aa

    return hook

def _save_input_offdiagonal(s_aa, ActHandler):
    def hook(module, input):
        a = ActHandler(input[0].data, module)
        print('shape of a is ', a.shape)
        # Initialize buffers
        if steps == 0:
            s_aa[module] = a
        else:
            s_aa[module] = torch.cat([s_aa[module], a], 0)
    return hook

def _save_grad_output(m_gg, GradHandler, batch_averaged=True):
    def hook(module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        gg = GradHandler(grad_output[0].data, module, batch_averaged)
        # Initialize buffers
        if steps == 0:
            m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(0))
        m_gg[module] += gg

    return hook

def _save_grad_output_offdiagonal(s_gg, GradHandler, batch_averaged=True):
    def hook(module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        g = GradHandler(grad_output[0].data, module, batch_averaged)
        # Initialize buffers
        if steps == 0:
            s_gg[module] = g
        else:
            s_gg[module] = torch.cat([s_gg[module], g], 0)
    return hook


def _prepare_model(model, m_aa, m_gg, s_aa=None, s_gg=None,
                   offdiagonal=False, proper=True, fix_layers=0):
    count = 0
    print(model)
    modules = []
    module_names = []
    print("=> We keep following layers in model. ")
    known_modules = {'Linear', 'Conv2d'}

    if proper:
        CovAHandler = ComputeCovA_proper()
        CovGHandler = ComputeCovG_proper()
        StoreAHandler = StoreA_proper()
        StoreGHandler = StoreG_proper()
    else:
        CovAHandler = ComputeCovA()
        CovGHandler = ComputeCovG()
        StoreAHandler = StoreA()
        StoreGHandler = StoreG()

    inp_hooks = [_save_input(m_aa, CovAHandler)]
    out_hooks = [_save_grad_output(m_gg, CovGHandler)]

    if offdiagonal:
        assert s_aa is not None
        assert s_gg is not None
        inp_hooks.append(_save_input_offdiagonal(s_aa, StoreAHandler))
        out_hooks.append(_save_grad_output_offdiagonal(s_gg, StoreGHandler))

    for module_name, module in model.named_modules():
        print(module)
        classname = module.__class__.__name__
        if classname in known_modules:
            modules.append(module)
            module_names.append(module_name)
            for inp_hook in inp_hooks:
                module.register_forward_pre_hook(inp_hook)
            for out_hook in out_hooks:
                module.register_backward_hook(out_hook)
            print('(%s): %s' % (count, module))
            count += 1
    modules = modules[fix_layers:]
    module_names = module_names[fix_layers:]
    return modules, module_names

def get_module_keys(model, prune_modules=None):
    modules = []
    known_modules = {'WrappedLayer'}
    for module_name, module in model.named_modules():
        classname = module.__class__.__name__
        if classname in known_modules:
            if hasattr(module, 'custom_name'):
                modules.append(module)

    return modules

def get_module_custom_names(model, prune_modules=None):
    custom_names = []
    known_modules = {'WrappedLayer'}
    for module_name, module in model.named_modules():
        classname = module.__class__.__name__
        if classname in known_modules:
            if hasattr(module, 'custom_name'):
                custom_names.append(module.custom_name)
    return custom_names

def _get_block_kro_dic(modules, s_dic):
    kro_dic = {}
    for i, mod_i in enumerate(modules):
        kro_dic[mod_i] = s_dic[mod_i].t() @ s_dic[mod_i]
    return kro_dic

def _get_kro_dic(modules, s_dic):
    kro_dic = {}
    for i, mod_i in enumerate(modules):
        for j, mod_j in enumerate(modules):
            kro_dic[f"{mod_i.custom_name}_{mod_j.custom_name}"] = s_dic[mod_i.custom_name].t() @ s_dic[mod_j.custom_name]
    return kro_dic

def _to_kro_dic(modules, m_dic):
    kro_dic = {}
    for i, mod_i in enumerate(modules):
        kro_dic[f"{mod_i.custom_name}_{mod_i.custom_name}"] = m_dic[mod_i.custom_name]
    return kro_dic

def convert_to_kro_dic(model, m_dic, prune_modules=None):
    custom_names = get_module_custom_names(model, prune_modules=prune_modules)
    print(custom_names, prune_modules)
    kro_dic = {}
    for i, custom_name in enumerate(custom_names):
        print(f"key is {custom_name}_{custom_name}")
        kro_dic[f"{custom_name}_{custom_name}"] = m_dic[custom_name]
    return kro_dic


def compare_norms(mat1, mat2):
    diff_norm = (mat1 - mat2).norm()
    print(f"Difference norm: {diff_norm.item()}, mat1 norm: {mat1.norm().item()}, mat2 norm: {mat1.norm().item()}.")
    print(f"Ratio of the norm difference to largest norm: {diff_norm.item()/max(mat1.norm(), mat2.norm()).item()}.")

def compare_blockwise_curvature_matrices(model, mat1, mat2):
    offset = 0
    total_norm = 0
    for param in model.parameters():
        num_param = param.numel()
        layer_norm = (mat1[offset:offset + num_param, offset:offset + num_param] -
                      mat2[offset:offset + num_param, offset:offset + num_param]).norm()
        print(f"norm for layer with params from {offset} to {offset + num_param} is ", layer_norm)
        total_norm += layer_norm ** 2
        offset += num_param
    total_norm = total_norm ** (1 / 2)
    print("Total norm difference across the layer blocks is ", total_norm.item())
    return total_norm

def are_blockwise_close(model, mat1, mat2, atol=1e-5):
    offset = 0
    for param in model.parameters():
        num_param = param.numel()
        print(f"checking for layer with params from {offset} to {offset + num_param}: ")
        print(torch.allclose(mat1[offset:offset + num_param, offset:offset + num_param],
                             mat2[offset:offset + num_param, offset:offset + num_param], atol=atol))
        offset += num_param


def inv_covs(xxt, ggt, eps, use_pi=True):
    # Modified from https://github.com/Thrandis/EKFAC-pytorch/blob/master/kfac.py

    """Inverses the covariances."""
    # num_locations is 1 for conv,
    # but for conv it is more like the product of spatial dimensions

    pi = 1.0
    if use_pi:
        # Computes pi
        tx = torch.trace(xxt) * ggt.shape[0]
        tg = torch.trace(ggt) * xxt.shape[0]
        pi = (tx / tg)

    # Regularizes and inverse

    diag_xxt = xxt.new(xxt.shape[0]).fill_((eps * pi) ** 0.5)
    diag_ggt = ggt.new(ggt.shape[0]).fill_((eps / pi) ** 0.5)
    print(diag_xxt.mean(), diag_ggt.mean())
    ixxt = (xxt + torch.diag(diag_xxt)).inverse()
    iggt = (ggt + torch.diag(diag_ggt)).inverse()
    return ixxt, iggt


def get_blockwise_kfac_inverse(model, m_aa=None, m_gg=None, s_aa=None, s_gg=None, num_samples=5000, damp=1e-5, use_pi=True, offload_cpu=False):
    num_params = sum([module.numel() for module in model.parameters()])
    device = list(model.parameters())[0].device

    if offload_cpu:
        emp_kfac_blockwise_fisher_inv = torch.zeros(num_params, num_params).cpu()
    else:
        emp_kfac_blockwise_fisher_inv = torch.zeros(num_params, num_params).to(device)

    block_offset = 0
    modules = get_module_keys(model)
    print(modules)
    if m_aa is None and m_gg is None and s_aa is not None and s_gg is not None:
        print("compute m_aa and m_gg again")
        m_aa, m_gg = _get_kro_dic(modules, s_aa), _get_kro_dic(modules, s_gg)

    for idx, mod in enumerate(modules):
        block_params = block_offset + kronecker_shape(m_aa[f"{mod.custom_name}_{mod.custom_name}"].shape, m_gg[f"{mod.custom_name}_{mod.custom_name}"].shape)
        print(f"updating diagonal fisher inverse block, {block_offset}:{block_params}")
        aainv, gginv = inv_covs(m_aa[f"{mod.custom_name}_{mod.custom_name}"]/num_samples, m_gg[f"{mod.custom_name}_{mod.custom_name}"]/num_samples, eps=damp, use_pi=use_pi)
        if offload_cpu:
            emp_kfac_blockwise_fisher_inv[block_offset:block_params, block_offset:block_params] = kronecker(
                gginv.cpu(), aainv.cpu())
        else:
            emp_kfac_blockwise_fisher_inv[block_offset:block_params, block_offset:block_params] = kronecker(
                gginv, aainv)

        block_offset += kronecker_shape(gginv.shape, aainv.shape)
        del aainv, gginv

    return emp_kfac_blockwise_fisher_inv

