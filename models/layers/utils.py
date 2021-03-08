"""
Implementing utility functional for layers
"""
import torch


def drop_connect(inputs, drop_rate, training):
    if not training: return inputs
    batch_size = inputs.size(0)
    keep_rates = (1 - drop_rate) * torch.ones(batch_size)[:,None,None,None]
    normalized_mask = torch.bernoulli(keep_rates) / keep_rates
    return inputs * normalized_mask

def swish(inputs):
    return inputs * torch.sigmoid(inputs)

def composite_swish(inputs_1, inputs_2):
    return inputs_1 * torch.sigmoid(inputs_2)


if __name__ == '__main__':
    x = torch.ones(5,1,1,1)
    print(drop_connect(x, 0.5, True))