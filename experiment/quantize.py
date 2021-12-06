"""Quantize a DNN"""

# pylint: disable=unspecified-encoding, logging-fstring-interpolation
# pylint: disable=invalid-name, too-many-locals, unused-variable
# pylint: disable=too-many-arguments, too-many-instance-attributes
# pylint: disable=relative-beyond-top-level

import logging
import json
import datetime
import time
from pathlib import Path

import torch
import torchvision.models.quantization
from torchvision.models import *

from .dnn import DNNExperiment
from .. import models
from .. import datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quantize")


class QuantizeExperiment(DNNExperiment):

    torchvision_constructors = {
        'googlenet': torchvision.models.quantization.googlenet,
        'mobilenet_v2': torchvision.models.quantization.mobilenet_v2,
    }

    def __init__(
            self,
            model_path: str,
            model_type: str,
            dataset: str,
            dl_kwargs: dict,
            train: bool,
            path: str,
            gpu: int = None,
            seed: int = None,
            debug: int = None,
    ):
        """
        Quantize a model.

        :param model_path: path to the model to quantize.
        :param model_type: the architecture of the model.
        :param dataset: the dataset on which to calibrate the model for quantization.
            Obviously, this should be the same dataset the model was trained on.
        :param dl_kwargs: the kwargs to pass to the dataloader for this dataset.
        :param train: if true, use the training dataset, else use the test dataset.
        :param path: the folder to save to.  See aicas.experiments.check_folder_structure
            for the folder schema.
        :param gpu: gpu to run on.
        :param seed: seed for random number generator.
        :param debug: if not None, it is an integer representing how many batches of
            data to run, instead of running the entire train/test dataset.
        """

        super().__init__(seed)

        self.resume = Path(model_path)
        self.model_path = self.resume

        self.model_type = model_type
        self.dataset = dataset
        self.train = train
        self.dl_kwargs = dl_kwargs
        self.path = path
        self.gpu = gpu
        self.device = None
        self.generate_uid()
        self.debug = debug
        self.save_path = None

    def build_quantized_model(self) -> None:
        """
        Build the model.  Note that this function calls self.to_device(), which
        sets self.device and transfers the model to self.device.
        """
        model = None

        if hasattr(models, self.model_type):
            model = getattr(models, model)(pretrained=pretrained)

        if self.model_type in self.torchvision_constructors:
            num_classes = datasets.num_classes[self.dataset]
            model_args = models.model_args(model)
            model = self.torchvision_constructors[self.model_type](pretrained=False, num_classes=num_classes, **model_args)

        if not model:
            raise ValueError(
                f"Model {model} not available in custom models or torchvision models"
            )

        resume = Path(self.model_path)
        assert resume.exists(), "Resume path does not exist"
        previous = torch.load(resume, map_location=torch.device('cpu'))
        model.load_state_dict(previous["model_state_dict"], strict=False)

        model.eval()

        return model

    def save_variables(self):
        """Return a dictionary of variables to save."""

        v = [
            "dataset",
            "model_path",
            "model_type",
            "path",
            "save_path"
        ]
        return {
            x: str(getattr(self, x))
            if isinstance(getattr(self, x), Path)
            else getattr(self, x)
            for x in v
        }

    def run(self, backend='fbgemm') -> None:
        """
        Quantize the model.

        Adapted from torchvision.models.quantization.utils.quantize_model
        """
        logger.info("Quantizing model ...")

        # generates self.train_dataset, self.val_dataset, self.train_dl, self.val_dl
        self.build_dataloader(self.dataset, **self.dl_kwargs)

        self.quantized_model = self.build_quantized_model()

        x, y = next(iter(self.train_dl))
        x, y = x.to(self.device), y.to(self.device)
        if backend not in torch.backends.quantized.supported_engines:
            raise RuntimeError("Quantized backend not supported ")
        torch.backends.quantized.engine = backend
        self.quantized_model.eval()
        # Make sure that weight qconfig matches that of the serialized models
        if backend == 'fbgemm':
            self.quantized_model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_per_channel_weight_observer)
        elif backend == 'qnnpack':
            self.quantized_model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_weight_observer)

        self.quantized_model.fuse_model()
        torch.quantization.prepare(self.quantized_model, inplace=True)
        self.quantized_model(x)
        torch.quantization.convert(self.quantized_model, inplace=True)

        logger.info("Quantization complete.")

        return self.quantized_model

    def save_run_information(self) -> None:
        """Save the parameters of the quantization to a json file."""

        quantize_folder = Path(self.path) / 'quantize'
        quantize_folder.mkdir(parents=True, exist_ok=True)

        # need to have a different name for each run in case of multiple
        # runs with different parameters
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        results_filename = f"quantize_results-{timestamp}.json"
        results_file = quantize_folder / results_filename

        with open(results_file, "w") as file:
            json.dump(self.save_variables(), file, indent=4)

    def save_model(self) -> None:
        """
        Save quantized model.
        """
        save_path = Path(self.path) / "quantize"
        save_path.mkdir(exist_ok=True, parents=True)
        self.save_path = save_path / f"quantized.pt"

        torch.save(
            {
                "model_state_dict": self.quantized_model.state_dict(),
            },
            save_path / f"quantized.pt",
        )
        self.save_run_information()

    def generate_uid(self):
        self.uid = "quantize"
        return self.uid
