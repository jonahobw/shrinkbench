import copy

from .magnitude import *

from ..pruning import (LayerPruning,
                       VisionPruning,
                       GradientMixin,
                       ActivationMixin,
                       AdversarialPruning,
                       AdversarialGradientMixin,
                       )
from .utils import (fraction_threshold,
                    fraction_mask,
                    map_importances,
                    flatten_importances,
                    importance_masks,
                    activation_importance,
                    )


class GreedyPGD(AdversarialGradientMixin, AdversarialPruning):

    default_pgd_args = {"eps": 2 / 255, "eps_iter": 0.001, "nb_iter": 10, "norm": np.inf}

    def __init__(self, model, dataloader, attack_kwargs, compression=1, device=None):
        attack_params = copy.deepcopy(self.default_pgd_args)
        attack_params.update(attack_kwargs)
        super().__init__(model=model, attack_name='pgd', dataloader=dataloader, attack_kwargs=attack_params, compression=compression, device=device)

    def model_masks(self, prunable=None):
        raise NotImplementedError("Class GreedyPGD is not a pruning method, it is inherited by other pruning "
                                  "methods.")


class GreedyPGDGlobalMagGrad(GreedyPGD):

    def model_masks(self):
        """Similar to GlobalMagGrad model_masks()"""
        params = self.params()
        grads = self.param_gradients(self.dl, self.attack, self.device)
        importances = {mod:
                           {p: params[mod][p] * grads[mod][p]
                            for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold, largest=True)
        return masks
