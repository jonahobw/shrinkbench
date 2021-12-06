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
                    map_zeros
                    )


class GreedyPGD(AdversarialGradientMixin, AdversarialPruning):

    default_pgd_args = {"eps": 2 / 255, "eps_iter": 0.001, "nb_iter": 10, "norm": np.inf}

    def __init__(self, model, dataloader, attack_kwargs, compression=1, device=None, debug=None):
        attack_params = copy.deepcopy(self.default_pgd_args)
        attack_params.update(attack_kwargs)
        super().__init__(model=model, attack_name='pgd', dataloader=dataloader, attack_kwargs=attack_params, compression=compression, device=device, debug=debug)

    def model_masks(self, prunable=None):
        raise NotImplementedError("Class GreedyPGD is not a pruning method, it is inherited by other pruning "
                                  "methods.")


class GreedyPGDGlobalMagGrad(GreedyPGD):

    def model_masks(self):
        """Similar to GlobalMagGrad model_masks()"""
        params = self.params()
        grads = self.param_gradients(dataloader=self.dl, attack=self.attack, device=self.device, batches=self.debug)

        # prune only the highest gradients wrt the loss
        importances = {mod:
                           {p: grads[mod][p]
                            for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction, largest=True)

        if threshold == 0:
            # there are too many 0 values in the tensor.  These 0 values need to be able
            # to be ranked.  To do this while maintaining the previous order of the tensor,
            # map 0 values to unique values

            importances = {mod:
                               {p: map_zeros(grads[mod][p])
                                for p in mod_params}
                           for mod, mod_params in params.items()}
            flat_importances = flatten_importances(importances)
            threshold = fraction_threshold(flat_importances, self.fraction, largest=True)

        return importance_masks(importances, threshold, largest=True, absolute=False)


class GreedyPGDGlobalMagGrad_param(GreedyPGD):

    def model_masks(self):
        """Similar to GlobalMagGrad model_masks()"""
        params = self.params()
        grads = self.param_gradients(dataloader=self.dl, attack=self.attack, device=self.device, batches=self.debug)

        # prune the highest gradient*parameter


        # prune only the highest gradients wrt the loss
        importances = {mod:
                           {p: params[mod][p] * grads[mod][p]
                            for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction, largest=True)

        if threshold == 0:
            # there are too many 0 values in the tensor.  These 0 values need to be able
            # to be ranked.  To do this while maintaining the previous order of the tensor,
            # map 0 values to unique values

            importances = {mod:
                               {p: map_zeros(params[mod][p] * grads[mod][p])
                                for p in mod_params}
                           for mod, mod_params in params.items()}
            flat_importances = flatten_importances(importances)
            threshold = fraction_threshold(flat_importances, self.fraction, largest=True)

        return importance_masks(importances, threshold, largest=True, absolute=False)


class GreedyPGDLayerMagGrad(LayerPruning, GreedyPGD):

    def layer_masks(self, module):
        params = self.module_params(module)
        grads = self.module_param_gradients(module, dataloader=self.dl, attack=self.attack, device=self.device, batches=self.debug)
        importances = {param: np.abs(grads[param]) for param, value in params.items()}
        masks = {param: fraction_mask(importances[param], self.fraction, largest=True, absolute=False)
                 for param, value in params.items() if value is not None}
        return masks