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
from ..metrics import correct
from ..util import OnlineStats

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
        save_imgs: bool = False,
        gpu: int = None,
        seed: int = None,
        debug: int = None,
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
        :param save_imgs: whether or not to save adversarial inputs.
        :param gpu: gpu to run on.
        :param seed: seed for random number generator.
        :param debug: if not None, it is an integer representing how many batches of
            data to attack, instead of attacking the entire train/test dataset.
        """

        super().__init__(seed)

        self.resume = Path(model_path)
        self.model_path = self.resume

        # generates self.model (torch.nn.module) class variable
        self.build_model(
            model_type, pretrained=False, resume=self.resume, dataset=dataset
        )

        # generates self.train_dataset, self.val_dataset, self.train_dl, self.val_dl
        self.build_dataloader(dataset, **dl_kwargs)

        self.model_type = model_type
        self.dataset = dataset
        self.train = train
        self.attack_name = attack.lower()
        self.attack = self.attack_constructors[self.attack_name]
        self.attack_params = attack_params
        self.path = path
        self.save_imgs = save_imgs
        self.gpu = gpu
        self.device = None
        self.runtime = None
        self.results = {}
        self.generate_uid()
        self.debug = debug

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
            "save_imgs",
            "runtime",
            "gpu",
            "results",
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
        self.to_device()
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

            if self.save_imgs:
                # todo implement this to be used in transfer attacks
                pass

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

        logger.info(results)
        self.results = results
        self.runtime = time.time() - since
        self.save_run_information()

    def save_run_information(self) -> None:
        """Save the parameters of the attack to a json file."""

        attack_folder = Path(self.path) / 'attacks' / self.attack_name
        attack_folder.mkdir(parents=True, exist_ok=True)

        # need to have a different name for each run because there might be multiple
        # runs of the same attack with different parameters
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_or_test = "train" if self.train else "test"
        results_file = (
            attack_folder / f"{train_or_test}_attack_results-{timestamp}.json"
        )

        with open(results_file, "w") as file:
            json.dump(self.save_variables(), file, indent=4)

    def generate_uid(self):
        self.uid = "attack"
        return self.uid
