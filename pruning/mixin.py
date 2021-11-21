""" Module with examples of common pruning patterns
"""
from .abstract import Pruning
from .utils import get_activations, get_param_gradients, get_adv_param_gradients


class ActivationMixin(Pruning):

    def update_activations(self):
        assert self.inputs is not None, \
            "Inputs must be provided for activations"
        self._activations = get_activations(self.model, self.inputs)

    def activations(self, only_prunable=True):
        if not hasattr(self, '_activations'):
            self.update_activations()
        if only_prunable:
            return {module: self._activations[module] for module in self.prunable}
        else:
            return self._activations

    def module_activations(self, module):
        if not hasattr(self, '_activations'):
            self.update_activations()
        return self._activations[module]


class GradientMixin(Pruning):

    def update_gradients(self):
        assert self.inputs is not None and self.outputs is not None, \
            "Inputs and Outputs must be provided for gradients"
        self._param_gradients = get_param_gradients(self.model, self.inputs, self.outputs)

    def param_gradients(self, only_prunable=True, attack_kwargs=None):
        if not hasattr(self, "_param_gradients"):
            self.update_gradients()
        if only_prunable:
            return {module: self._param_gradients[module] for module in self.prunable}
        else:
            return self._param_gradients

    def module_param_gradients(self, module):
        if not hasattr(self, "_param_gradients"):
            self.update_gradients()
        return self._param_gradients[module]

    def input_gradients(self):
        raise NotImplementedError("Support coming soon")

    def output_gradients(self):
        raise NotImplementedError("Support coming soon")


class AdversarialGradientMixin(GradientMixin):

    def update_gradients(self, dataloader, attack, device=None, batches=None):
        assert dataloader is not None, "Dataloader must be passed for adversarial gradients over whole dataset."
        self._param_gradients = get_adv_param_gradients(model=self.model, dl=dataloader, attack=attack, device=device, batches=batches)

    def param_gradients(self, dataloader=None, attack=None, device=None, only_prunable=True, batches=None):
        if not hasattr(self, "_param_gradients"):
            assert attack is not None, "Attack must be provided to compute adversarial gradients."
            assert dataloader is not None, "Dataloader must be provided to compute adversarial gradients."
            self.update_gradients(dataloader=dataloader, attack=attack, device=device, batches=batches)
        if only_prunable:
            return {module: self._param_gradients[module] for module in self.prunable}
        else:
            return self._param_gradients

    def module_param_gradients(self, dataloader=None, attack=None, device=None, only_prunable=True, batches=None):
        if not hasattr(self, "_param_gradients"):
            assert attack is not None, "Attack must be provided to compute adversarial gradients."
            assert dataloader is not None, "Dataloader must be provided to compute adversarial gradients."
            self.update_gradients(dataloader=dataloader, attack=attack, device=device, batches=batches)
        return self._param_gradients[module]