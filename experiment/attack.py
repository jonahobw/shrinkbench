"""Run an attack on a DNN"""

# pylint: disable=unspecified-encoding, logging-fstring-interpolation
# pylint: disable=invalid-name, too-many-locals, unused-variable
# pylint: disable=too-many-arguments, too-many-instance-attributes
# pylint: disable=relative-beyond-top-level

import logging
import json
import datetime
import time
from pathlib import Path

from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

# from cleverhans.plot.pyplot_image import pair_visual  # might implement later
from tqdm import tqdm

from .dnn import DNNExperiment
from ..metrics import correct, both_correct
from ..util import OnlineStats
from .quantize import QuantizeExperiment

logger = logging.getLogger("attack")


class AttackExperiment(DNNExperiment):
    """
    Run an attack on a model.
    """

    attack_constructors = {"pgd": projected_gradient_descent}

    def __init__(
        self,
        model_path: str,
        model_type: str,
        dataset: str,
        dl_kwargs: dict,
        train: bool,
        attack: str,
        attack_params: dict,
        path: str,
        transfer_model_path: str = None,
        gpu: int = None,
        seed: int = None,
        debug: int = None,
        quantized: bool = False,
        pre_quantized_model: str=None,
    ):
        """
        Run an attack on a model.

        :param model_path: path to the model to attack.
        :param model_type: the architecture of the model.
        :param dataset: the dataset on which to run the attack.  Obviously, this should be the same
            dataset the model was trained on.
        :param dl_kwargs: the kwargs to pass to the dataloader for this dataset.
        :param train: if true, use the training dataset, else use the test dataset.
        :param attack: the attack method.
        :param attack_params: parameters to pass to the attack implementation.
        :param path: the 'attack' folder to save to.  See aicas.experiments.check_folder_structure
            for the folder schema.
        :param transfer_model_path: the path of the transfer model.
        :param gpu: gpu to run on.
        :param seed: seed for random number generator.
        :param debug: if not None, it is an integer representing how many batches of
            data to attack, instead of attacking the entire train/test dataset.
        :param quantized: if true, the provided model path is to a quantized model.  must
            also provide pre_quantized model.
        :param pre_quantized_model: a path to the float version of the model (with float
            architecture, not the quantized architecture).  The adversarial inputs will
            be generated on this model and the quantized model's accuracy will be assesed.
        """

        super().__init__(seed)

        self.resume = Path(model_path)
        self.model_path = self.resume

        self.model_type = model_type
        self.dataset = dataset
        self.train = train
        self.attack_name = attack.lower()
        self.attack = self.attack_constructors[self.attack_name]
        self.attack_params = attack_params
        self.path = path
        self.gpu = gpu
        self.device = None
        self.results = None
        self.transfer_results = None
        self.transfer_model_path = None
        self.generate_uid()
        self.debug = debug
        self.quantized = quantized
        self.pre_quantized_model = pre_quantized_model

        if not self.quantized:
            # generates self.model (torch.nn.module) class variable
            self.build_model(
                model_type, pretrained=False, resume=self.resume, dataset=dataset
            )
            self.surrogate_model = None
        else:
            self.surrogate_model = self.build_model(
                model_type, pretrained=False, resume=self.pre_quantized_model, dataset=dataset
            )
            self.model = QuantizeExperiment.load_quantized_model(
                self.model_path,
                self.model_type,
                self.dataset,
            )

        if transfer_model_path:
            self.transfer_model_path = Path(transfer_model_path)
            self.transfer_model = self.build_model(model_type, pretrained=False, resume=self.transfer_model_path, dataset=self.dataset, set=False)
            self.transfer_model.to(self.device)

        # generates self.train_dataset, self.val_dataset, self.train_dl, self.val_dl
        self.build_dataloader(dataset, **dl_kwargs)

    def save_variables(self):
        """Return a dictionary of variables to save."""

        v = [
            "attack_name",
            "attack_params",
            "dataset",
            "model_path",
            "model_type",
            "train",
            "path",
            "gpu",
            "results",
            "transfer_results",
            "transfer_model_path",
            "quantized",
            "pre_quantized_model",
        ]
        return {
            x: str(getattr(self, x))
            if isinstance(getattr(self, x), Path)
            else getattr(self, x)
            for x in v
        }

    def run(self) -> None:
        """
        Run the attack.

        Code adapted from cleverhans tutorial
        https://github.com/cleverhans-lab/cleverhans/blob/master/tutorials/torch/cifar10_tutorial.py
        """
        since = time.time()
        self.model.eval()

        time.sleep(0.1)

        if self.train:
            dl = self.train_dl
            data = "train"
        else:
            dl = self.val_dl
            data = "test"

        results = {"inputs_tested": 0}

        clean_acc1 = OnlineStats()
        clean_acc5 = OnlineStats()
        adv_acc1 = OnlineStats()
        adv_acc5 = OnlineStats()

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(f"{self.attack_name} on {data} dataset")

        for i, (x, y) in enumerate(epoch_iter, start=1):
            x, y = x.to(self.device), y.to(self.device)
            if self.quantized:
                x_adv = self.attack(self.surrogate_model, x, **self.attack_params)
                x_adv, x, y = x_adv.to('cpu'), x.to('cpu'), y.to('cpu')
                self.model.to('cpu')
            else:
                x_adv = self.attack(self.model, x, **self.attack_params)
            y_pred = self.model(x)  # model prediction on clean examples
            y_pred_adv = self.model(x_adv)  # model prediction on adversarial examples

            results["inputs_tested"] += y.size(0)

            clean_c1, clean_c5 = correct(y_pred, y, (1, 5))
            clean_acc1.add(clean_c1 / dl.batch_size)
            clean_acc5.add(clean_c5 / dl.batch_size)

            adv_c1, adv_c5 = correct(y_pred_adv, y, (1, 5))
            adv_acc1.add(adv_c1 / dl.batch_size)
            adv_acc5.add(adv_c5 / dl.batch_size)

            epoch_iter.set_postfix(
                clean_acc1=clean_acc1.mean,
                clean_acc5=clean_acc5.mean,
                adv_acc1=adv_acc1.mean,
                adv_acc5=adv_acc5.mean,
            )

            if self.debug is not None and i == self.debug:
                break

        results["clean_acc1"] = clean_acc1.mean
        results["clean_acc5"] = clean_acc5.mean
        results["adv_acc1"] = adv_acc1.mean
        results["adv_acc5"] = adv_acc5.mean

        results["runtime"] = time.time() - since

        logger.info(results)
        self.results = results
        self.save_run_information()

    def run_transfer(self) -> None:
        """
        Run a transfer attack.

        Code adapted from cleverhans tutorial
        https://github.com/cleverhans-lab/cleverhans/blob/master/tutorials/torch/cifar10_tutorial.py
        """
        topk = (1, 5)

        since = time.time()
        self.model.eval()
        self.transfer_model.to(self.device)
        self.transfer_model.eval()

        time.sleep(0.1)

        if self.train:
            dl = self.train_dl
            data = "train"
        else:
            dl = self.val_dl
            data = "test"

        results = {"inputs_tested": 0,
                   "both_correct1": 0, "both_correct5": 0,
                   "transfer_correct1": 0, "transfer_correct5": 0,
                   "target_correct1": 0, "target_correct5": 0}

        transf_acc1 = OnlineStats()
        transf_acc5 = OnlineStats()
        targ_acc1 = OnlineStats()
        targ_acc5 = OnlineStats()

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(f"Transfer attack {self.attack_name} on {data} dataset")

        for i, (x, y) in enumerate(epoch_iter, start=1):
            x, y = x.to(self.device), y.to(self.device)

            self.transfer_model.to(self.device)

            # generate adversarial examples using the transfer model
            x_adv_transfer = self.attack(self.transfer_model, x, **self.attack_params)

            if self.quantized:
                x_adv_transfer, y = x_adv_transfer.to('cpu'), y.to('cpu')
                self.model.to('cpu')
                self.transfer_model.to('cpu')
            # get predictions from transfer and target model
            y_pred_transfer = self.transfer_model(x_adv_transfer)
            y_pred_target = self.model(x_adv_transfer)

            results["inputs_tested"] += y.size(0)

            adv_c1_transfer, adv_c5_transfer = correct(y_pred_transfer, y, (1, 5))
            results["transfer_correct1"] += adv_c1_transfer
            results["transfer_correct5"] += adv_c5_transfer
            transf_acc1.add(adv_c1_transfer / dl.batch_size)
            transf_acc5.add(adv_c5_transfer / dl.batch_size)

            adv_c1_target, adv_c5_target = correct(y_pred_target, y, (1, 5))
            results["target_correct1"] += adv_c1_target
            results["target_correct5"] += adv_c5_target
            targ_acc1.add(adv_c1_target / dl.batch_size)
            targ_acc5.add(adv_c5_target / dl.batch_size)
            
            both_correct1, both_correct5 = both_correct(y_pred_transfer, y_pred_target, y, (1, 5))
            results["both_correct1"] += both_correct1
            results["both_correct5"] += both_correct5

            epoch_iter.set_postfix(
                transf_acc1=transf_acc1.mean,
                transf_acc5=transf_acc5.mean,
                targ_acc1=targ_acc1.mean,
                targ_acc5=targ_acc5.mean,
            )

            if self.debug is not None and i == self.debug:
                break

        results["both_correct1"] = results["both_correct1"] / results["inputs_tested"]
        results["both_correct5"] = results["both_correct5"] / results["inputs_tested"]

        results["transfer_correct1"] = results["transfer_correct1"] / results["inputs_tested"]
        results["transfer_correct5"] = results["transfer_correct5"] / results["inputs_tested"]

        results["target_correct1"] = results["target_correct1"] / results["inputs_tested"]
        results["target_correct5"] = results["target_correct5"] / results["inputs_tested"]

        results["transfer_runtime"] = time.time() - since
        results["transfer_model"] = str(self.transfer_model_path)

        logger.info(results)
        self.transfer_results = results
        self.save_run_information(transfer=True)

    def save_run_information(self, transfer=False) -> None:
        """Save the parameters of the attack to a json file."""

        attack_folder = Path(self.path) / 'attacks' / self.attack_name
        attack_folder.mkdir(parents=True, exist_ok=True)

        # need to have a different name for each run because there might be multiple
        # runs of the same attack with different parameters
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_or_test = "train" if self.train else "test"
        results_filename = f"{train_or_test}_attack_results-{timestamp}.json"
        results_file = attack_folder / results_filename

        if transfer:
            transfer_folder = attack_folder / "transfer_attack"
            transfer_folder.mkdir(parents=True, exist_ok=True)
            results_file = transfer_folder / results_filename

        with open(results_file, "w") as file:
            json.dump(self.save_variables(), file, indent=4)

    def generate_uid(self):
        self.uid = "attack"
        return self.uid
