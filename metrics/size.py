"""Model size metrics
"""

import numpy as np
import torch
from . import nonzero, dtype2bits


def model_size(model, as_bits=False):
    """Returns absolute and nonzero model size

    Arguments:
        model {torch.nn.Module} -- Network to compute model size over

    Keyword Arguments:
        as_bits {bool} -- Whether to account for the size of dtype

    Returns:
        int -- Total number of weight & bias params
        int -- Out total_params exactly how many are nonzero
    """

    total_params = 0
    nonzero_params = 0
    for tensor in model.parameters():
        t = np.prod(tensor.shape)
        nz = nonzero(tensor.detach().cpu().numpy())
        if as_bits:
            bits = dtype2bits[tensor.dtype]
            t *= bits
            nz *= bits
        total_params += t
        nonzero_params += nz
    if total_params == 0:
        # probably a quantized model, use model.state_dict() instead of model.parameters()
        nonzero_params = 0

        for key in model.state_dict():
            tensor = model.state_dict()[key]
            if not isinstance(tensor, torch.Tensor):
                continue
            t = np.prod(tensor.shape)
            nz = torch.sum(tensor.detach().cpu() != 0)
            if as_bits:
                bits = dtype2bits[tensor.dtype]
                t *= bits
                nz *= bits
            total_params += t
            nonzero_params += nz
    return int(total_params), int(nonzero_params)
