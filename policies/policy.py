"""
This is a base class for:
* Pruners
* Optimizers

Each type subclasses PolicyBase by another unifying class
and has a separate config_reader.

"""

import torch

class PolicyBase(object):
    def __init__(self, *args, **kwargs):
        pass

    def on_epoch_begin(self, *args, **kwargs):
        pass

    def on_minibatch_begin(self, *args, **kwargs):
        pass

    def before_backward_pass(self, *args, **kwargs):
        pass

    def before_parameter_optimization(self, *args, **kwargs):
        pass

    def on_parameter_optimization(self, *args, **kwargs):
        pass

    def after_parameter_optimization(self, *args, **kwargs):
        pass

    def on_minibatch_end(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        pass
