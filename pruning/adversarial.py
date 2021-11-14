import torch.nn as nn
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from .vision import VisionPruning


class AdversarialPruning(VisionPruning):

    attack_constructors = {"pgd": projected_gradient_descent}

    def __init__(self, model, attack_name, dataloader, attack_kwargs, compression=1, device=None):
        self.device = device
        x, y = next(iter(dataloader))
        if device:
            x, y = x.to(self.device), y.to(self.device)
        super().__init__(model, inputs=x, outputs=y, compression=compression)
        self.dl = dataloader
        self.attack_name = attack_name
        self.attack_kwargs = attack_kwargs

    def attack(self, inputs):
        attack_fn = self.attack_constructors[self.attack_name]
        return attack_fn(self.model, inputs, **self.attack_kwargs)
