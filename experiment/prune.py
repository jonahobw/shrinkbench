"""Pruning a DNN."""

# pylint: disable=relative-beyond-top-level, too-many-arguments, invalid-name
# pylint: disable=unspecified-encoding, too-many-locals
import copy
import json
from typing import Callable
from .train import TrainingExperiment
from .. import strategies
from ..pruning import AdversarialPruning
from ..metrics import model_size, flops
from ..util import printc


class PruningExperiment(TrainingExperiment):
    """Prune a DNN."""

    def __init__(
        self,
        dataset: str,
        model: str,
        strategy: str,
        compression: int,
        seed: int = None,
        path: str = None,
        dl_kwargs: {} = None,
        train_kwargs: {} = None,
        debug: bool = False,
        pretrained: bool = False,
        resume: str = None,
        resume_optim: bool = False,
        save_freq: int = 10,
        checkpoint_metric: str = None,
        early_stop_method: str = None,
        lr_schedule: Callable = None,
        gpu: int = None,
        save_one_checkpoint: bool=False,
        attack_kwargs: {} = None,
    ):
        """
        Setup class variabls.

        :param dataset: dataset to train on.
        :param model: the architecture of the model.
        :param strategy: the pruning strategy.
        :param compression: the desired ratio of parameters in original to pruned model.
        :param seed: the random seed for uid generation.
        :param path: the path to the folder where results will be stored.
        :param dl_kwargs: args for the dataloader.
        :param train_kwargs: args for training, including number of epochs, optimizer type,
            learning rate, and weight decay.
        :param debug: whether or not to run in debug mode.
        :param pretrained: whether or not to use a pretrained model.
        :param resume: path to a pretrained model.
        :param resume_optim: whether or not to resume from a previous optimizer.
        :param save_freq: how many epochs to wait between saving models (best model seen so far
            will still be saved regardless of this parameter).
        :param checkpoint_metric: metric to use to figure out which is the best model seen
            during training.
        :param early_stop_method: method for determining early training termination.
        :param lr_schedule: function with signature (epoch) which returns the learning rate
            for that epoch.
        :param gpu: the number of the gpu to run on.
        :param save_one_checkpoint: if true, removes all previous checkpoints and only keeps this one.
            Since each checkpoint may be hundreds of MB, this saves lots of memory.
        :param attack_kwargs: specified when doing adversarial pruning; the parameters for the attack to
            generate adversarial inputs.  Note the attack is from the CleverHans library
        """

        super().__init__(
            dataset,
            model,
            seed,
            path,
            dl_kwargs,
            train_kwargs,
            debug,
            pretrained,
            resume,
            resume_optim,
            save_freq,
            checkpoint_metric,
            early_stop_method,
            lr_schedule,
            gpu,
            save_one_checkpoint,
        )
        self.add_params(strategy=strategy, compression=compression, attack_kwargs=attack_kwargs)

        self.apply_pruning(strategy, compression, attack_kwargs)

        self.save_freq = save_freq
        self.metrics = None

    def apply_pruning(self, strategy: str, compression: int, attack_kwargs: {} = None) -> None:
        """Apply the pruning to the model."""
        printc("Pruning model ...", color="GREEN")
        self.to_device()
        constructor = getattr(strategies, strategy)
        if issubclass(constructor, AdversarialPruning):
            assert attack_kwargs is not None, f"Attack kwargs must be provided for {strategy} pruning."
            copy_attack_kwargs = copy.deepcopy(attack_kwargs)
            train = copy_attack_kwargs.pop('train')
            dl = self.train_dl if train else self.val_dl
            self.pruning = constructor(model=self.model, dataloader=dl, attack_kwargs=copy_attack_kwargs, compression=compression, device=self.device, debug=self.debug)
        else:
            x, y = next(iter(self.train_dl))
            self.pruning = constructor(self.model, x, y, compression=compression)
        self.pruning.apply()
        printc("Pruning Completed", color="GREEN")

    def run(self) -> None:
        """Run the finetuning and metrics on the model."""
        self.freeze()
        printc(f"Running {repr(self)}", color="YELLOW")
        self.to_device()
        self.build_logging(self.train_metrics, self.path)

        self.save_metrics()

        if self.pruning.compression > 1:
            self.run_epochs()

    def save_metrics(self) -> None:
        """Save the pruning stats."""
        self.metrics = self.pruning_metrics()
        with open(self.path / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=4)
        printc(json.dumps(self.metrics, indent=4), color="GRASS")
        summary = self.pruning.summary()
        summary_path = self.path / "masks_summary.csv"
        summary.to_csv(summary_path)

    def pruning_metrics(self) -> {}:
        """Collect the pruning metrics."""
        metrics = {}
        # Model Size
        size, size_nz = model_size(self.model)
        metrics["size"] = size
        metrics["size_nz"] = size_nz
        metrics["compression_ratio"] = size / size_nz

        x, y = next(iter(self.val_dl))
        x, y = x.to(self.device), y.to(self.device)

        # FLOPS
        ops, ops_nz = flops(self.model, x)
        metrics["flops"] = ops
        metrics["flops_nz"] = ops_nz
        metrics["theoretical_speedup"] = ops / ops_nz

        # Accuracy
        loss, acc1, acc5 = self.run_epoch(False, -1)
        self.log_epoch(-1)

        metrics["loss"] = loss
        metrics["val_acc1"] = acc1
        metrics["val_acc5"] = acc5

        return metrics

    def generate_uid(self):
        self.uid = "prune"
        return self.uid
