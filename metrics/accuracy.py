import numpy as np
import torch
from torch.backends import cudnn
from tqdm import tqdm
import random


def correct(output, target, topk=(1,)):
    """Computes how many correct outputs with respect to targets

    Does NOT compute accuracy but just a raw amount of correct
    outputs given target labels. This is done for each value in
    topk. A value is considered correct if target is in the topk
    highest values of output.
    The values returned are upperbounded by the given batch size

    [description]

    Arguments:
        output {torch.Tensor} -- Output prediction of the model
        target {torch.Tensor} -- Target labels from data

    Keyword Arguments:
        topk {iterable} -- [Iterable of values of k to consider as correct] (default: {(1,)})

    Returns:
        List(int) -- Number of correct values for each topk
    """

    with torch.no_grad():
        maxk = max(topk)
        # Only need to do topk for highest k, reuse for the rest
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res


def accuracy(model, dataloader, topk=(1,), seed=None, loss_func=None, debug=None):
    """Compute accuracy/loss of a model over a dataloader for various topk

    Arguments:
        model {torch.nn.Module} -- Network to evaluate
        dataloader {torch.utils.data.DataLoader} -- Data to iterate over

    Keyword Arguments:
        topk {iterable} -- [Iterable of values of k to consider as correct] (default: {(1,)})
        loss_func {torch.nn._Loss} -- function to compute the loss
        seed {int} -- if provided, sets the random seeds of python, pytorch, and numpy

    Returns:
        List(float) -- List of accuracies for each topk
    """

    if seed is not None:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        random.seed(seed)

        # Numpy
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)

    # Use same device as model
    device = next(model.parameters()).device

    model.eval()

    accs = np.zeros(len(topk))
    total_tested = 0
    loss = 0

    running_accs = {}
    for x in topk:
        running_accs[f"top{x}"] = 0

    epoch_iter = tqdm(dataloader)
    epoch_iter.set_description("Model Accuracy")

    with torch.no_grad():

        for i, (input, target) in enumerate(epoch_iter):
            if debug is not None and i > debug:
                break
            input = input.to(device)
            target = target.to(device)
            output = model(input)

            accs += np.array(correct(output, target, topk))
            if loss_func:
                loss += loss_func(output, target).item()
            total_tested += len(input)
            for i, x in enumerate(topk):
                running_accs[f"top{x}"] = accs[i]/total_tested
            epoch_iter.set_postfix(
                **running_accs
            )

    # print(f"Total inputs tested: {total_tested}")
    # print(f"Total correct: {accs}")

    # Normalize over data length
    loss /= total_tested
    accs /= total_tested

    if loss_func:
        accs = np.append(accs, loss)

    return accs
