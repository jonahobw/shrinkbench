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
import torchvision.models
from tqdm import tqdm

from .dnn import DNNExperiment
from .. import models
from .. import datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quantize")


class QuantizeExperiment(DNNExperiment):

    torch_quantization_constructors = {
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
            backend = 'fbgemm',
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
        if backend not in torch.backends.quantized.supported_engines:
            raise RuntimeError("Quantized backend not supported ")
        self.backend = backend

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

    def build_float_model(self, load_state_dict=True) -> None:
        """
        Build the floating point model.
        """
        model = None

        if hasattr(models, self.model_type):
            model = getattr(models, self.model_type)(pretrained=False, quantized=True)

        if self.model_type in self.torch_quantization_constructors:
            num_classes = datasets.num_classes[self.dataset]
            model_args = models.model_args(model)
            model = self.torch_quantization_constructors[self.model_type](pretrained=False, num_classes=num_classes, **model_args)

        if model is None:
            raise ValueError(
                f"Model {model} not available in custom models or torchvision models"
            )

        if load_state_dict:
            model = self.load_state_dict(model, self.model_path)

        model.eval()
        return model

    def load_state_dict(self, model, path):
        resume = Path(path)
        assert resume.exists(), "Resume path does not exist"
        previous = torch.load(resume, map_location=torch.device('cpu'))
        model.load_state_dict(previous["model_state_dict"], strict=False)
        return model

    def prepare_for_quantization(self, model):
        torch.backends.quantized.engine = self.backend
        model.eval()
        # Make sure that weight qconfig matches that of the serialized models
        if self.backend == 'fbgemm':
            model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_per_channel_weight_observer)
        elif self.backend == 'qnnpack':
            model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_weight_observer)

        model.fuse_model()
        torch.quantization.prepare(model, inplace=True)
        return model

    def quantize(self, prepped_model, dataloader=None, batches=None):
        if dataloader is not None:
            # calibrate model
            dl_iter = tqdm(dataloader)
            dl_iter.set_description("Calibrating model for quantization")
            for i, (x, y) in enumerate(dl_iter):
                prepped_model(x)
                if batches is not None and i == batches:
                    break
        torch.quantization.convert(prepped_model, inplace=True)
        return prepped_model


    def run(self, load_path=None) -> None:
        """
        Quantize the model.

        If load_path is provided, then load an existing quantized model.
        """
        logger.info("Quantizing model ...")

        prepped_model = self.prepare_for_quantization(self.build_float_model(load_state_dict=load_path is None))

        # generates self.train_dataset, self.val_dataset, self.train_dl, self.val_dl
        self.build_dataloader(self.dataset, **self.dl_kwargs)
        dl = self.train_dl if self.train else self.val_dl
        self.quantized_model = self.quantize(prepped_model=prepped_model,
                                             dataloader=dl,
                                             batches=self.debug)

        if load_path:
            self.load_state_dict(self.quantized_model, load_path)

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
        return self.save_path

    def generate_uid(self):
        self.uid = "quantize"
        return self.uid
