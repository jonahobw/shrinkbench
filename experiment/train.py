"""Training a DNN."""

# pylint: disable=relative-beyond-top-level, too-many-instance-attributes
# pylint: disable=invalid-name, too-many-arguments, too-many-locals
# pylint: disable=no-member

import time
import json
from typing import Callable

import torch
from torch import nn
from tqdm import tqdm

from .dnn import DNNExperiment
from ..metrics import correct
from ..util import printc, OnlineStats


class TrainingExperiment(DNNExperiment):
    """Training a DNN."""

    default_dl_kwargs = {"batch_size": 128, "pin_memory": False, "num_workers": 8}

    default_train_kwargs = {
        "optim": "SGD",
        "epochs": 30,
        "lr": 1e-3,
    }

    def __init__(
        self,
        dataset: str,
        model: str,
        seed: int = None,
        path: str = None,
        dl_kwargs: {} = None,
        train_kwargs: {} = None,
        debug: bool = False,
        pretrained: bool = False,
        resume: str = None,
        resume_optim: bool = False,
        save_freq: int = 1000,
        checkpoint_metric: str = None,
        early_stop_method: str = None,
        lr_schedule: Callable = None,
        gpu: int = None,
        save_one_checkpoint: bool = False,
    ) -> None:

        """
        Setup class variables.

        :param dataset: dataset to train on.
        :param model: the architecture of the model.
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
        :param save_one_checkpoint: if true, removes all previous checkpoints and only keeps
            one at a time. Since each checkpoint may be hundreds of MB, this saves lots of memory.
        """

        # Default children kwargs
        super().__init__(seed)
        dl_kwargs = {**self.default_dl_kwargs, **dl_kwargs}
        train_kwargs = {**self.default_train_kwargs, **train_kwargs}

        self.resume = resume
        self.path = path
        self.save_freq = save_freq
        self.debug = debug

        # the metric that will be used to find the best model
        self.checkpoint_metric = checkpoint_metric
        # value of the best model metric
        self.best_metrics = None
        self.gpu = gpu
        self.device = None # gets set when calling self.build_model()
        self.early_stop_method = early_stop_method
        self.save_one_checkpoint = save_one_checkpoint

        params = locals()
        params["dl_kwargs"] = dl_kwargs
        params["train_kwargs"] = train_kwargs
        params.pop("lr_schedule")
        if path is not None:
            params["path"] = str(path)
        self.add_params(**params)
        # Save params

        self.build_dataloader(dataset, **dl_kwargs)

        self.build_model(model, pretrained, resume, dataset)

        self.build_train(
            resume_optim=resume_optim, lr_schedule=lr_schedule, **train_kwargs
        )

    def check_best_metric(self, metrics: {}) -> bool:
        """
        Returns true if a model metric is the best seen so far.
        Uses self.checkpoint_metric for comparison.

        :param metrics: dictionary of model metrics from the last epoch.

        :return: True if metric is the best seen so far, else False
        """

        # function has not been called yet, set the best metric to worst case
        if self.best_metrics is None:
            self.best_metrics = metrics
            return True

        # compare the new metric value with the best seen and update state if new metric is better
        if (
            "loss" in self.checkpoint_metric
            and metrics[self.checkpoint_metric]
            < self.best_metrics[self.checkpoint_metric]
        ):
            self.best_metrics = metrics
            return True
        if (
            "acc" in self.checkpoint_metric
            and metrics[self.checkpoint_metric]
            > self.best_metrics[self.checkpoint_metric]
        ):
            self.best_metrics = metrics
            return True

        return False

    def run(self) -> None:
        """Run the training."""

        self.freeze()
        printc(f"Running {repr(self)}", color="YELLOW")
        self.build_logging(self.train_metrics, self.path)
        time.sleep(1)
        self.run_epochs()

    def build_train(
        self,
        optim: str,
        epochs: int,
        lr_schedule: Callable,
        resume_optim: bool = False,
        **optim_kwargs,
    ) -> None:
        """Generate optimizer and learning rate scheduler."""

        default_optim_kwargs = {
            "SGD": {"momentum": 0.9, "nesterov": True, "lr": 1e-3},
            "Adam": {"momentum": 0.9, "betas": (0.9, 0.99), "lr": 1e-4},
        }

        self.epochs = epochs

        # Optim
        if isinstance(optim, str):
            constructor = getattr(torch.optim, optim)
            if optim in default_optim_kwargs:
                optim_kwargs = {**default_optim_kwargs[optim], **optim_kwargs}
            optim = constructor(self.model.parameters(), **optim_kwargs)

        self.optim = optim

        if resume_optim:
            assert hasattr(self, "resume"), "Resume must be given for resume_optim"
            previous = torch.load(self.resume)
            self.optim.load_state_dict(previous["optim_state_dict"])

        # Assume classification experiment
        self.loss_func = nn.CrossEntropyLoss()

        # set up the optimizer to adjust learning rate during training
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_schedule)

    def checkpoint(self, epoch: int) -> None:
        """
        Save model parameters from last epoch.

        If self.save_one_checkpoint is true, removes all previous checkpoints and only keeps this
        one. Since each checkpoint may be hundreds of MB, this saves lots of memory.
        """

        checkpoint_path = self.path / "checkpoints"
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        if self.save_one_checkpoint:
            for checkpoint in checkpoint_path.glob("*"):
                checkpoint.unlink()
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optim_state_dict": self.optim.state_dict(),
            },
            checkpoint_path / f"checkpoint-{epoch}.pt",
        )

    def run_epochs(self) -> None:
        """Run the training epochs."""

        since = time.time()
        try:
            for epoch in range(1, self.epochs + 1):
                loss, acc1, acc5 = self.train(epoch)
                val_loss, val_acc1, val_acc5 = self.eval(epoch)

                metrics = {
                    "train_loss": loss,
                    "train_acc1": acc1,
                    "train_acc5": acc5,
                    "val_loss": val_loss,
                    "val_acc1": val_acc1,
                    "val_acc5": val_acc5,
                }
                # Checkpoint epochs if it is the best model seen so far or if save frequency
                # is triggered
                if self.checkpoint_metric is not None and self.check_best_metric(
                    metrics
                ):
                    print(
                        f"\n{self.checkpoint_metric} ({metrics[self.checkpoint_metric]}) "
                        f"improved over previous best, checkpointing model.\n"
                    )
                    self.checkpoint(epoch)
                elif epoch % self.save_freq == 0:
                    print("Save frequency met, checkpointing model")
                    self.checkpoint(epoch)

                self.log(timestamp=time.time() - since)
                self.log_epoch(epoch)

                # update the learning rate (must be done once per epoch, or once per
                #   training + testing iteration, that is why this is done in this
                #   function rather than self.run_epoch())
                self.lr_scheduler.step()

                # optionally stop early
                if self.early_stop_method and self.early_stop(metrics):
                    break

                if self.debug is not None:
                    break

        except KeyboardInterrupt:
            printc(f"\nInterrupted at epoch {epoch}. Tearing Down", color="RED")

        print("Training ended, checkpointing last model.")
        self.checkpoint(epoch)

    def run_epoch(self, train: bool, epoch: int = 0) -> tuple[int]:
        """Run a single epoch."""

        if train:
            self.model.train()
            prefix = "train"
            dl = self.train_dl
        else:
            prefix = "val"
            dl = self.val_dl
            self.model.eval()

        total_loss = OnlineStats()
        acc1 = OnlineStats()
        acc5 = OnlineStats()
        step_size = OnlineStats()
        step_size.add(self.lr_scheduler.get_last_lr()[0])

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(f"{prefix.capitalize()} Epoch {epoch if train else '1'}/{self.epochs if train else '1'}")

        with torch.set_grad_enabled(train):
            for i, (x, y) in enumerate(epoch_iter, start=1):
                if self.debug is not None and i > self.debug:
                    break
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = self.loss_func(yhat, y)
                if train:
                    loss.backward()

                    self.optim.step()
                    self.optim.zero_grad()

                c1, c5 = correct(yhat, y, (1, 5))
                total_loss.add(loss.item() / dl.batch_size)
                acc1.add(c1 / dl.batch_size)
                acc5.add(c5 / dl.batch_size)

                epoch_iter.set_postfix(
                    loss=total_loss.mean,
                    top1=acc1.mean,
                    top5=acc5.mean,
                    step_size=step_size.mean,
                )

        self.log(
            **{
                f"{prefix}_loss": total_loss.mean,
                f"{prefix}_acc1": acc1.mean,
                f"{prefix}_acc5": acc5.mean,
                "lr": self.lr_scheduler.get_last_lr()[0],
            }
        )

        return total_loss.mean, acc1.mean, acc5.mean

    def train(self, epoch: int = 0) -> tuple[int]:
        """Run a single training epoch."""
        return self.run_epoch(True, epoch)

    def eval(self, epoch: int = 0) -> tuple[int]:
        """Run a single validation epoch."""
        return self.run_epoch(False, epoch)

    @property
    def train_metrics(self) -> list:
        """Generate the training metrics to be logged to a csv."""
        return [
            "epoch",
            "timestamp",
            "train_loss",
            "train_acc1",
            "train_acc5",
            "val_loss",
            "val_acc1",
            "val_acc5",
            "lr",
        ]

    def generate_uid(self):
        self.uid = "train"
        return self.uid

    def __repr__(self) -> str:
        """Dunders."""
        if not isinstance(self.params["model"], str) and isinstance(
            self.params["model"], torch.nn.Module
        ):
            self.params["model"] = self.params["model"].__module__

        assert isinstance(
            self.params["model"], str
        ), f"\nUnexpected model inputs: {self.params['model']}"
        return json.dumps(self.params, indent=4)
