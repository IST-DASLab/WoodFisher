'''
Example command:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate.py \
    --from_checkpoint_path ../exp_root/exp_str_prune_\(mobilenet\)_gradmask_\(0.7528\)_epochs_\(100\)_\(woodfisherblock\)_fit_\(10000\)_fisher_\(400\)_\(2400\)_final_all_layers_best_v4/20200711_00-33-58_802839_20/best_checkpoint.ckpt \
    --gpus=0,1,2,3 --not-oldfashioned
'''

from policies.trainers import build_training_policy_from_config


from utils.utils import (preprocess_for_device, recompute_bn_stats, get_total_sparsity)

from utils.checkpoints import load_checkpoint, save_checkpoint, get_unwrapped_model
from utils.flop_utils import get_flops
from utils.manager_utils import my_test, my_test_dataset, get_dataloaders
from utils.masking_utils import get_wrapped_model
from utils.parse_config import read_config
from utils.pruner_utils import pruner_epoch_end_functional, pruner_after_parameter_optimization_functional

import math
import torch
import torch.onnx
import time
import os
import logging
from options import get_parser
from main import setup_logging

DEFAULT_TRAINER_NAME = "default_trainer"


def get_args():
    args = get_parser()
    args = setup_logging(args)

    args = preprocess_for_device(args)
    args.topk = True

    if 'tao' in args.arch:
        args.tao_augm = True
    else:
        args.tao_augm = False

    args.old_fashioned = not args.not_oldfashioned
    args.disable_log_soft = True
    return args

def export_onnx(args, model):
    # model = get_unwrapped_model(model)
    x = torch.randn(args.batch_size, 3, 224, 224, requires_grad=True).to(args.device)
    # torch_out = model(x)

    if args.onnx_nick:
        onnx_nick = args.onnx_nick
    else:
        onnx_nick = 'resnet_pruned.onnx'

        # Export the model
    torch.onnx.export(model.module,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      onnx_nick,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])
    print("ONNX EXPORT COMPLETED. EXITING")




if __name__ == '__main__':

    args = get_args()

    if args.deterministic:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(args)
    if args.from_checkpoint_path is not None:

        epoch, model, optimizer, lr_scheduler, model_config = load_checkpoint(args.from_checkpoint_path, current_model_config=None,
                          config_dict=None, trainer_name=None, old_fashioned=args.old_fashioned, args=args, return_model_config=True)
    else:
        raise NotImplementedError


    args.arch = model_config['arch']
    args.dset = model_config['dataset']
    args.use_butterfly = model_config['use_butterfly']

    data = (args.dset, args.dset_path)

    if args.device.type == 'cuda':
        # if len(args.gpus) > 1:
        if not args.no_dataparallel:
            model = torch.nn.DataParallel(model, device_ids=args.gpus)
            model.to(args.device)
        else:
            model = model.to(args.device)

        # if not args.export_onnx:
        #     model = get_wrapped_model(model)

    if args.config_path is None:
        args.config_path = "./configs/resnet50_imagenet_woodfisher_90_all.yaml" # default config file
        assert os.path.isfile(args.config_path)

    config = args.config_path if isinstance(args.config_path, dict) else read_config(args.config_path)

    trainers = [build_training_policy_from_config(model, config, trainer_name=DEFAULT_TRAINER_NAME)]

    _, test_loader = get_dataloaders(args, data)
    ckpt_acc, ckpt_loss, _ = my_test(args, model, log_dict=None, test_loader=test_loader)
    print("The test accuracy and loss of the checkpoint are ", ckpt_acc, ckpt_loss)

    sparsity_dicts = pruner_epoch_end_functional(model)
    print("The sparsity dicts are ", sparsity_dicts)

    if args.mask_onnx:

        device_before = next(model.named_parameters())[1].device
        model.to('cpu')
        # model = get_wrapped_model(model)

        pruner_after_parameter_optimization_functional(model=model)

        model = get_unwrapped_model(model)
        model.to(device_before)

        sparsity_dicts = pruner_epoch_end_functional(model)

        _, test_loader = get_dataloaders(args, data)
        masked_acc, masked_loss, _ = my_test(args, model, log_dict=None, test_loader=test_loader)
        print("The masked checkpoint accuracy and loss are ", masked_acc, masked_loss)

        if args.save_dense_also:
            save_checkpoint(epoch, model_config, model, trainers[0].optimizer,
                            trainers[0].lr_scheduler, args.run_dir, is_best=False, nick='sparsified')

    if args.export_onnx:
        export_onnx(args, model)

    if args.flops:
        total_flops, module_flops, module_names = get_flops(args, model)