"""Base class for dnn experiments."""

from typing import Union
import pathlib

import torch
import torchvision.models
from torch.utils.data import DataLoader
from torch.backends import cudnn

from .base import Experiment
from .. import datasets
from .. import models
from ..models.head import mark_classifier
from ..util import printc


class DNNExperiment(Experiment):

    def __init__(self, seed: int=42):
        super().__init__(seed=seed)
        self.gpu = None


    def to_device(self) -> None:
        """Move model to CPU/GPU."""

        # Torch CUDA config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.gpu is not None:
            try:
                self.device = torch.device(f"cuda:{self.gpu}")
            except AssertionError as e:
                printc(e, color="ORANGE")
        if not torch.cuda.is_available():
            printc("GPU NOT AVAILABLE, USING CPU!", color="ORANGE")
        self.model.to(self.device)
        cudnn.benchmark = True  # For fast training.

    def build_dataloader(self, dataset: str, **dl_kwargs) -> None:
        """Build the dataloader."""

        constructor = getattr(datasets, dataset)
        self.train_dataset = constructor(train=True)
        self.val_dataset = constructor(train=False)
        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **dl_kwargs)
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, **dl_kwargs)

    def build_model(
        self,
        model: Union[str, torch.nn.Module],
        pretrained: bool = True,
        resume: str = None,
        dataset: str = None,
    ) -> None:
        """Build the model."""

        if isinstance(model, str):
            if hasattr(models, model):
                model = getattr(models, model)(pretrained=pretrained)

            elif hasattr(torchvision.models, model):
                # need dataset to know number of classes
                assert dataset is not None, (
                    "If model is not in shrinkbench, 'dataset' argument must be"
                    " passed to self.build_model so the number of classes is known."
                )
                num_classes = datasets.num_classes[dataset]
                model_args = models.model_args(model)
                model = getattr(torchvision.models, model)(
                    pretrained=pretrained, num_classes=num_classes, **model_args
                )
                mark_classifier(model)  # add is_classifier attribute
            else:
                raise ValueError(
                    f"Model {model} not available in custom models or torchvision models"
                )

        self.model = model

        if resume is not None:
            self.resume = pathlib.Path(self.resume)
            assert self.resume.exists(), "Resume path does not exist"
            previous = torch.load(self.resume)
            self.model.load_state_dict(previous["model_state_dict"], strict=False)
