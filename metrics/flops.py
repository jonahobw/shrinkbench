import numpy as np
import torch
from torch import nn

from . import nonzero
from .abstract_flops import dense_flops, conv2d_flops
from ..pruning.utils import get_activations
from ..pruning import Conv2dMasked, LinearMasked


def _conv2d_flops(module, activation):
    # Auxiliary func to use abstract flop computation

    # Drop batch & channels. Channels can be dropped since
    # unlike shape they have to match to in_channels
    input_shape = activation.shape[2:]
    # TODO Add support for dilation and padding size
    return conv2d_flops(in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        input_shape=input_shape,
                        kernel_shape=module.kernel_size,
                        padding=module.padding_mode,
                        strides=module.stride,
                        dilation=module.dilation)


def _linear_flops(module, activation):
    # Auxiliary func to use abstract flop computation
    return dense_flops(module.in_features, module.out_features)


def flops(model, input, quantized=False):
    """Compute Multiply-add FLOPs estimate from model

    Arguments:
        model {torch.nn.Module} -- Module to compute flops for
        input {torch.Tensor} -- Input tensor needed for activations
        quantized {bool} -- Whether or not the passed model is quantized

    Returns:
        tuple:
        - int - Number of total FLOPs
        - int - Number of FLOPs related to nonzero parameters
    """
    if quantized:
        print("WARNING: FLOPS CALCULATION NOT DEBUGGED FOR QUANTIZED MODELS.")

    # todo:
    # known issues - additions for identity layer in ResNet are not accounted for. (LambdaLayer)

    FLOP_fn = {
        nn.Conv2d: _conv2d_flops,
        nn.Linear: _linear_flops,
        Conv2dMasked: _conv2d_flops,
        LinearMasked: _linear_flops,
        nn.quantized.modules.conv.Conv2d: _conv2d_flops,
        nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d: _conv2d_flops,
        nn.quantized.modules.linear.Linear: _linear_flops
    }

    total_flops = nonzero_flops = 0.0
    activations = get_activations(model, input, quantized=quantized)

    # The ones we need for backprop
    for m, (act, _) in activations.items():
        if m.__class__ in FLOP_fn:
            if not quantized:
                w = m.weight.detach().cpu().numpy().copy()
            else:
                w = m.weight().detach().cpu().clone()
            module_flops = FLOP_fn[m.__class__](m, act)
            total_flops +=  float(module_flops)
            # For our operations, all weights are symmetric so we can just
            # do simple rule of three for the estimation
            if not quantized:
                nz = float(nonzero(w).sum())
            else:
                nz = float(torch.sum(w != 0))
            nonzero_flops += float(module_flops) * nz / float(np.prod(w.shape))

    return total_flops, nonzero_flops
