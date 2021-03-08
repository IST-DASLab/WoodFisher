import os
import errno
import torch
import shutil
import logging
import inspect
from models.mlpnet import MlpNet

from models import get_model # from __init__.py
from utils.masking_utils import is_wrapped_layer, WrappedLayer, get_wrapped_model

__all__ = ['save_eval_checkpoint', 'save_checkpoint', 
           'load_eval_checkpoint', 'load_checkpoint', 'get_unwrapped_model']


# Add dummy stuff here if you need to add an optimizer or a lr_scheduler
# that have required constructor arguments (and you want to recover it 
# from state dict later and need some dummy value)
DUMMY_VALUE_FOR_OPT_ARG = {'lr': 1e-3, 'gamma': 0.9}


def should_unwrap_layer(layer: 'nn.Module') -> bool:
    return isinstance(layer, WrappedLayer)

def unwrap_module(module: 'nn.Module', prefix='.'):
    """
    Recursive function which iterates over WRAPPED_MODULES of this
    module and unwraps them.
    """
    # print("before unwrap", module)
    module_dict = dict(module.named_children())
    for name, sub_module in module_dict.items():
        if should_unwrap_layer(sub_module):
            setattr(module, name, sub_module.unwrap())
            # print(prefix, name)
            print(f'Module {prefix + name} was successfully unwrapped')
            continue
        unwrap_module(sub_module, prefix + name + '.')
    # print("after unwrap", module)

def get_unwrapped_model(model: 'nn.Module') -> 'nn.Module':
    """
    Function which unwrappes the wrapped layers of received model.
    """
    unwrap_module(model)
    return model

def save_eval_checkpoint(model_config: str, model: 'nn.Module', checkpoint_path: str):
    """
    Save the model state dict with all layer unwrapped and 
    pruning masks applied.
    
    Arguments:
        model_config {dict} -- {'arch': arch, 'dataset': dataset}
        path {str} -- path to save wrapped model (e.g.: exps_root/sample_run/run_id)
    """
    if not os.path.isdir(checkpoint_path):
        raise IOError(errno.ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(checkpoint_path))

    model = get_unwrapped_model(model)
    if isinstance(model, torch.nn.DataParallel):
        logging.debug('Was using data parallel')
        model = model.module
    model_state_dict = model.state_dict()
    state_dict = dict()
    state_dict['model_config'] = model_config
    state_dict['model_state_dict'] = model_state_dict
    torch.save(state_dict, os.path.join(checkpoint_path, 'eval_ready_state_dict.ckpt'))

def load_eval_checkpoint(checkpoint_path: str) -> 'nn.Module':
    """
    Load the evaluation ready model given the chepoint path.
    """
    try:
        state_dict = torch.load(os.path.join(checkpoint_path, 'eval_ready_state_dict.ckpt'))
    except:
        raise IOError(errno.ENOENT, 'Evaluation checkpoint does not exist at', os.path.abspath(checkpoint_path))
    model_config = state_dict['model_config']
    model = get_model(model_config['arch'], model_config['dataset'])
    model.load_state_dict(state_dict['model_state_dict'])
    return model

def save_checkpoint(epoch, model_config, model, optimizer, lr_scheduler,
                    checkpoint_path: str, is_best=False, nick=''):
    """
    This function damps the full checkpoint for the running manager.
    Including the epoch, model, optimizer and lr_scheduler states. 
    """
    if not os.path.isdir(checkpoint_path):
        raise IOError(errno.ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(checkpoint_path))

    if is_best:
        logging.info(f"saving best checkpoint at epoch {epoch}")

    checkpoint_dict = dict()
    checkpoint_dict['epoch'] = epoch
    checkpoint_dict['model_config'] = model_config
    checkpoint_dict['model_state_dict'] = model.state_dict()
    checkpoint_dict['optimizer'] = {
        'type': type(optimizer),
        'state_dict': optimizer.state_dict()
    }
    checkpoint_dict['lr_scheduler'] = {
        'type': type(lr_scheduler),
        'state_dict': lr_scheduler.state_dict()
    }
    if nick == '':
        ckpt_suffix = ''
    else:
        ckpt_suffix = '_' + nick

    path_regular = os.path.join(checkpoint_path, f'regular_checkpoint{ckpt_suffix}.ckpt')
    path_best = os.path.join(checkpoint_path, f'best_checkpoint{ckpt_suffix}.ckpt')
    torch.save(checkpoint_dict, path_regular)
    if is_best:
        shutil.copyfile(path_regular, path_best)


def _load_optimizer(cls, state_dict, model):
    try:
        varnames, _, _, defaults = inspect.getargspec(cls.__init__)
        vars_needing_vals = varnames[2:-len(defaults)]
        kwargs = {v: DUMMY_VALUE_FOR_OPT_ARG[v] for v in vars_needing_vals}
        optimizer = cls(model.parameters(), **kwargs)
    except KeyError as e:
        logging.debug(f"You need to add a dummy value for {e} to DUMMY_VALUE_FOR_OPT_ARG dictionary in checkpoints module.")
        raise
    optimizer.load_state_dict(state_dict)
    return optimizer

def _load_lr_scheduler(cls, state_dict, optimizer):
    try:
        varnames, _, _, defaults = inspect.getargspec(cls.__init__)
        # print(defaults, 'this is defaults')
        # print(cls, 'this is cls\n')
        vars_needing_vals = varnames[2:-len(defaults)]
        kwargs = {v: DUMMY_VALUE_FOR_OPT_ARG[v] for v in vars_needing_vals}
        lr_scheduler = cls(optimizer, **kwargs)
    except KeyError as e:
        logging.debug(f"You need to add a dummy value for {e} to DUMMY_VALUE_FOR_OPT_ARG dictionary in checkpoints module.")
        raise
    lr_scheduler.load_state_dict(state_dict)
    return lr_scheduler

def update_from_old_fashioned_checkpoint(state_dict, checkpoint_dict):
    """
    loads the wts and biases of the layers onto this model, and ignores the previous models!
    """
    checkpoint_keys = checkpoint_dict.keys()
    print("these are the checkpoint keys", checkpoint_keys)
    processed_keys = {}
    for key in checkpoint_keys:

        if 'module' in key:
            name_pieces = key.split('.')
            assert 'module' == name_pieces[0]
            subname = '.'.join(name_pieces[1:])
            firstname = name_pieces[0] + '.'
        else:
            subname = key
            firstname = ''

        if len(subname.split('.')) == 4 or len(subname.split('.')) == 3:
            subname = subname.replace('._layer', '')
        elif len(subname.split('.')) != 2:
            raise NotImplementedError

        assert len(subname.split('.')) == 2
        # assume the previous checkpoints has format like fc1.weight
        layer, param_type = subname.split('.')
        if subname in processed_keys:
            print("repeated, are they same?", (checkpoint_dict[processed_keys[subname]] == checkpoint_dict[key]).all())
            assert (checkpoint_dict[processed_keys[subname]] == checkpoint_dict[key]).all()
        else:
            processed_keys[subname] = key

        if 'mask' in key:
            final_name = layer + '.' + param_type
        else:
            final_name = layer + '._layer.' + param_type
        print(f"Copy param {key} in checkpoint to {final_name}")

        state_dict[final_name] = checkpoint_dict[key]

    return state_dict

def update_from_dan_checkpoint(state_dict, checkpoint_dict):
    """
    loads the wts and biases of the layers onto this model, and ignores the previous models!
    """
    checkpoint_keys = checkpoint_dict.keys()
    for key in checkpoint_keys:
        name_pieces = key.split('.')
        subname = '.'.join(name_pieces[1:])
        if subname in state_dict:
            if state_dict[subname].shape != checkpoint_dict[key].shape:
                print(f"Changing to the shape of state_dict[{subname}], which is {state_dict[subname].shape}")
                checkpoint_dict[key] = checkpoint_dict[key].view(state_dict[subname].shape)
            state_dict[subname] = checkpoint_dict[key]
            print(f"Copy param {key} with shape {checkpoint_dict[key].shape} in checkpoint to {subname}")
        else:
            name_pieces = subname.split('.')
            idx = 0
            while idx < len(name_pieces):
                if name_pieces[idx] == 'weight' or name_pieces[idx] == 'bias':
                    break
                idx += 1

            if idx == len(name_pieces):
                print(f"parameter with name: {key} does not exist in the model")
            else:
                aux_pieces = name_pieces[0:idx] + ['_layer'] + name_pieces[idx:]
                aux_subname = '.'.join(aux_pieces)
                if aux_subname in state_dict:
                    if state_dict[aux_subname].shape != checkpoint_dict[key].shape:
                        print(f"Changing to the shape of state_dict[{aux_subname}], which is {state_dict[aux_subname].shape}")
                        checkpoint_dict[key] = checkpoint_dict[key].view(state_dict[aux_subname].shape)
                    print(f"Copy param {key} with shape {checkpoint_dict[key].shape} in checkpoint to {aux_subname}")
                    state_dict[aux_subname] = checkpoint_dict[key]
                else:
                    if subname.split('.')[0] == 'linear':
                        print('auxsubname and subname are ', aux_subname, subname)
                        state_dict['fc._layer.' + subname.split('.')[1]] = checkpoint_dict[key]
                        new_name = 'fc._layer.' + subname.split('.')[1]
                        print(f"Copy param {key} with shape {checkpoint_dict[key].shape} in checkpoint to {new_name}")
                    else:
                        print(f"parameter with name: {key} does not exist in the model")


    return state_dict


def test_mlp_checkpoint_acc(args, checkpoint_dict):
    mlp = MlpNet(None, 'mnist')
    mlp.load_state_dict(checkpoint_dict)
    mlp.to(args.device)
    from policies.manager import my_test
    print("Directly testing vanilla MlpNet with my_test routine")
    my_test(args, mlp)


def update_cifarnet_checkpoint(state_dict, checkpoint_dict):
    checkpoint_keys = checkpoint_dict.keys()
    print("these are the checkpoint keys", checkpoint_keys)
    for key in checkpoint_keys:
        name_pieces = key.split('.')
        assert len(name_pieces) == 2
        layer_type = name_pieces[0]
        param_type = name_pieces[1]
        state_dict[f'{layer_type}._layer.{param_type}'] = checkpoint_dict[key]
    return state_dict

def load_checkpoint(full_checkpoint_path: str, current_model_config=None, config_dict=None, trainer_name=None,
                    old_fashioned=False, ckpt_epoch=-1, args=None, unwrapped=False, return_model_config=False):
    """
    Loads checkpoint give full checkpoint path.

    config_dict corresponds to the config present in yaml
    current_model_config and config_dict do the ground work related to checkpointing from scratch if needed

    old fashioned: load parameters from a model whose layers may not have been wrapped!
    """
    try:

        if 'str' in full_checkpoint_path.lower():
            checkpoint_dict = torch.load(full_checkpoint_path, map_location='cuda:0')
        else:
            checkpoint_dict = torch.load(full_checkpoint_path)
    except:
        raise IOError(errno.ENOENT, 'Checkpoint file does not exist at', os.path.abspath(full_checkpoint_path))

    from policies.trainers import build_optimizer_from_config, build_lr_scheduler_from_config

    try:
        print('current_model_config', current_model_config)
        if 'model_config' in checkpoint_dict:
            model_config = checkpoint_dict['model_config']
        elif current_model_config is not None:
            model_config = current_model_config
        else:
            raise NotImplementedError

        if not unwrapped:
            model = get_wrapped_model(get_model(*model_config.values()))
        else:
            model = get_model(*model_config.values())

        updated_state_dict = model.state_dict()

        print("updated_state_dict", updated_state_dict.keys())
        print("args is ", args)

        if args is not None:
            if args.arch == 'simplenet' or args.arch == 'cifarnet':
                state_dict = checkpoint_dict
            else:
                if 'state_dict' in checkpoint_dict:
                    state_dict = checkpoint_dict['state_dict']
                elif 'model_state_dict' in checkpoint_dict:
                    state_dict = checkpoint_dict['model_state_dict']
                else:
                    raise NotImplementedError

        if old_fashioned:
            updated_state_dict = update_from_old_fashioned_checkpoint(updated_state_dict, state_dict)
        else:
            if not args.arch == 'cifarnet':
                # in resnet there is --not-old_fashioned flag enabled
                updated_state_dict = update_from_dan_checkpoint(updated_state_dict, state_dict)
            else:
                updated_state_dict = update_cifarnet_checkpoint(updated_state_dict, state_dict)
        model.load_state_dict(updated_state_dict)

        if trainer_name is not None:
            trainer_dict = config_dict['trainers'][trainer_name]
            optimizer = build_optimizer_from_config(model, trainer_dict['optimizer'])

            lr_scheduler, _ = build_lr_scheduler_from_config(optimizer, trainer_dict['lr_scheduler'])
        else:
            optimizer = None
            lr_scheduler = None

        if 'epoch' in checkpoint_dict:
            checkpoint_epoch = checkpoint_dict['epoch']
        else:
            checkpoint_epoch = ckpt_epoch
    except Exception as e:
        raise TypeError(f'Checkpoint file is not valid. {e}')

    if return_model_config:
        return checkpoint_epoch, model, optimizer, lr_scheduler, model_config
    else:
        return checkpoint_epoch, model, optimizer, lr_scheduler

