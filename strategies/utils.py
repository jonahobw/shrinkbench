"""Auxiliary heuristics for choosing parameters

These functions are implemented as purely numpy functions for ease
of debugging and interpretation. They are then plugged into the
rest of the framework pipeline
"""

import numpy as np


def fraction_threshold(tensor, fraction, largest=False):
    """Compute threshold quantile for a given scoring function

    Given a tensor and a fraction of parameters to keep,
    computes the quantile so that only the specified fraction
    are larger than said threshold after applying a given scoring
    function. By default, magnitude pruning is applied so absolute value
    is used.

    Arguments:
        tensor {numpy.ndarray} -- Tensor to compute threshold for
        fraction {float} -- Fraction of parameters to keep
        largest {bool} -- if false, returns the quantile so that only the
            specified fraction are **larger** than said threshold, if
            true, returns the quantile so that only the specified fraction
            are **smaller** than said threshold

    Returns:
        float -- Threshold
    """
    assert isinstance(tensor, np.ndarray)
    if largest:
        return np.quantile(tensor, fraction)
    threshold = np.quantile(tensor, 1 - fraction)
    return threshold


def threshold_mask(tensor, threshold, absolute=True, largest=False):
    """Given a fraction or threshold, compute binary mask

    Arguments:
        tensor {numpy.ndarray} -- Array to compute the mask for

    Keyword Arguments:
        threshold {float} -- Absolute threshold for dropping params
        largest {bool} -- If false, masks the smallest absolute values.  If
                        true, masks the largest actual value.  (used for
                        adversarial pruning)
        absolute {bool} -- If true, then mask values by absolute value.  If
                        false, then mask all values by actual value.

    Returns:
        np.ndarray -- Binary mask
    """
    assert isinstance(tensor, np.ndarray)

    # idx is set to 1 for parameters to remove
    if absolute and not largest:
        idx = np.logical_and(tensor < threshold, tensor > -threshold)
    if absolute and largest:
        idx = np.logical_or(tensor > threshold, tensor < -threshold)
    if not absolute and largest:
        idx = tensor > threshold
    if not absolute and not largest:
        idx = tensor < threshold
    mask = np.ones_like(tensor)
    mask[idx] = 0
    return mask


def fraction_mask(tensor, fraction, largest=False, absolute=True):
    assert isinstance(tensor, np.ndarray)
    threshold = fraction_threshold(tensor, fraction, largest=largest)
    if threshold == 0:
        # there are too many 0 values in the tensor.  These 0 values need to be able
        # to be ranked.  To do this while maintaining the previous order of the tensor,
        # map 0 values to unique values
        tensor = map_zeros(tensor)
        threshold = fraction_threshold(tensor, fraction, largest=largest)
    return threshold_mask(tensor, threshold, largest=largest, absolute=absolute)


def map_zeros(tensor):
    """
    Maps 0 values in the tensor to unique values while keeping the relative ordering
    of the tensor.

    This is helpful when you are pruning a tensor and lots of values in the tensor are
    0.  In this case, pruning only the values less than 0 might not be enough to get to
    a certain level of prune compression.

    Maps each element in the tensor through the function
            f(x) = {    x,                  x < 0,
                        uniform[0, 0.5],    x = 0,
                        x+1,                x > 0.
    """
    return (
        np.multiply(tensor, tensor < 0)
        + np.multiply(1 + tensor, tensor > 0)
        + np.multiply(
            np.random.uniform(0, 0.5, size=tensor.shape).astype(tensor.dtype),
            tensor == 0,
        )
    )


def flatten_importances(importances):
    return np.concatenate(
        [
            importance.flatten()
            for _, params in importances.items()
            for _, importance in params.items()
        ]
    )


def map_importances(fn, importances):
    return {
        module: {param: fn(importance) for param, importance in params.items()}
        for module, params in importances.items()
    }


def importance_masks(importances, threshold, largest=False, absolute=True):
    """

    :param importances:
    :param threshold:
    :param largest: bool to indicate whether or not to remove the largest weights or smallest.
    :return:
    """
    return map_importances(
        lambda imp: threshold_mask(imp, threshold, largest=largest, absolute=absolute),
        importances,
    )


def norms_tensor(tensor, ord, matrix_mode=False):
    if matrix_mode:
        assert len(tensor.shape) > 2
        tensor_flattened = tensor.reshape(*tensor.shape[:2], -1)
    else:
        tensor_flattened = tensor.reshape(tensor.shape[0], -1)
    norms = []
    for w in tensor_flattened:
        norms.append(np.linalg.norm(w, ord))
    return np.array(norms)


def activation_importance(weight, activation, norm=1):
    activation = activation.mean(axis=0)  # reduce batch dim
    norms = norms_tensor(activation, norm)
    # add filter dims
    if len(weight.shape) > 2:
        # normalize spatial dims
        norms /= activation.shape[1] * activation.shape[2]
        # TODO refactor with np.tile or similar
        norms = norms[..., np.newaxis, np.newaxis]
        norms = np.repeat(norms, weight.shape[2], axis=1)
        norms = np.repeat(norms, weight.shape[3], axis=2)
        # normalize across spatial dimensions

    if len(weight.shape) > len(norms.shape):
        # broadcasting will take care of it
        return weight * norms
    else:
        # for bias
        return weight * norms.mean()


# def largest_norm_channel_mask(tensor, fraction, ord=1, matrix_mode=False):
#     # Assume channels is first axis, true for conv & linear
#     n_channels = tensor.shape[0]
#     norms = norms_tensor(tensor, ord, matrix_mode)
#     idx = np.argsort(-norms)
#     pruned_channels = idx[int(n_channels*fraction):]
#     mask = np.ones_like(tensor)
#     mask[pruned_channels, ...] = 0.0
#     return mask


# def largest_norm_activation_mask(tensor, activation, fraction, ord=1, matrix_mode=False):
#     # Sum over batch dimnesion
#     activation = activation.sum(axis=0)
#     n_channels = tensor.shape[0]
#     norms = norms_tensor(activation, ord, matrix_mode)
#     idx = np.argsort(-norms)
#     prunned_channels = idx[int(n_channels*fraction):]
#     mask = np.ones_like(tensor)
#     mask[prunned_channels, ...] = 0.0
#     return mask
