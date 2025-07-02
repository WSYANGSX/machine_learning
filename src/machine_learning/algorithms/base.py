from typing import Literal, Mapping, Any, Union

import os
import yaml
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from machine_learning.models import BaseNet
from machine_learning.types.aliases import FilePath
from machine_learning.utils.others import print_dict, print_segmentation


class AlgorithmBase(ABC):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        models: Mapping[str, BaseNet],
        data: Mapping[str, Union[Dataset, Any]],
        name: str | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ):
        """Base class of all algorithms

        Args:
            cfg (YamlFilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg dict.
            models (Mapping[str, BaseNet]): Models required by the algorithm, {"net1": BaseNet1, "net2": BaseNet2}.
            data (Mapping[str, Union[Dataset, Any]]): Parsed specific dataset data, must including train dataset and val
            dataset, may contain data information of the specific dataset.
            name (str, optional): Name of the algorithm. Defaults to None.
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
        """
        super().__init__()

        self._models = models
        self._optimizers = {}
        self._schedulers = {}
        self.training = False

        # ------------------------ configure device -----------------------
        self._device = self._configure_device(device)

        # -------------------- load algo configuration --------------------
        self._cfg = self._load_config(cfg)
        self._validate_config()

        self.batch_size = self.cfg["data_loader"].get("batch_size", 256)
        self.mini_batch_size = self.batch_size // self.cfg["data_loader"].get("subdevision", 1)

        # ---------------------- configure algo name ----------------------
        self._name = name if name is not None else self._cfg.get("algorithm", {}).get("name", __class__.__name__)

        # -------------------- configure models of algo -------------------
        self._configure_models()
        if self.cfg["model"]["initialize_weights"]:
            self._initialize_weights()

        # --------------------- configure optimizers ----------------------
        self._configure_optimizers()

        # --------------------- configure schedulers ----------------------
        self._configure_schedulers()

        # ------------------------ configure data -------------------------
        self._initialize_dependent_on_data(data)

    @property
    def name(self) -> str:
        return self._name

    @property
    def models(self) -> dict[str, BaseNet]:
        return self._models

    @property
    def optimizers(self) -> dict[str, BaseNet]:
        return self._optimizers

    @property
    def schedulers(self) -> dict[str, BaseNet]:
        return self._schedulers

    @property
    def cfg(self) -> dict:
        return self._cfg

    @property
    def device(self) -> torch.device:
        return self._device

    def _configure_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_config(self, config: FilePath | Mapping[str, Any]) -> dict:
        if isinstance(config, Mapping):
            cfg = config
        else:
            if not (os.path.splitext(config)[1] == ".yaml" or os.path.splitext(config)[1] == ".yml"):
                raise ValueError("Input path is not a yaml file path.")
            with open(config, "r") as f:
                cfg = yaml.safe_load(f)

        print_segmentation()
        print("Configuration parameters: ")
        print_dict(cfg)
        print_segmentation()

        return cfg

    def _validate_config(self):
        """Validate the config of the algorithm"""
        required_sections = ["algorithm", "model", "optimizer", "scheduler", "data_loader"]
        for section in required_sections:
            if section not in self.cfg:
                raise ValueError(f"The necessary parts are missing in the configuration file: {section}.")

    def _configure_models(self):
        """Configure models of the algo and show their structures"""
        for model in self._models.values():
            model.to(self._device)
            model.view_structure()

    def _initialize_weights(self) -> None:
        """Initialize the weight parameters of models"""
        for model in self._models.values():
            model._initialize_weights()

    def _initialize_dependent_on_data(self, data: Mapping[str, Union[Dataset, Any]]) -> None:
        """Initialize the training、validation(test) data loaders and other dataset-specific parameters.

        Args:
            data (Mapping[str, Union[Dataset, Any]]): Parsed specific dataset data, must including train dataset and val
            dataset, may contain data information of the specific dataset.
        """
        self.cfg["data"] = {}

        _necessary_key_type_couples_ = {"train_dataset": Dataset, "val_dataset": Dataset}
        for key, type in _necessary_key_type_couples_.items():
            if key not in data or not isinstance(data[key], type):
                raise ValueError(f"Input data mapping has no {key} or {key} is not Dataset type.")

        train_dataset, val_dataset = data.pop("train_dataset"), data.pop("val_dataset")

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=self.cfg["data_loader"].get("data_shuffle", True),
            num_workers=self.cfg["data_loader"].get("num_workers", 4),
            collate_fn=train_dataset.collate_fn if hasattr(train_dataset, "collate_fn") else None,
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.mini_batch_size,
            shuffle=False,
            num_workers=self.cfg["data_loader"].get("num_workers", 4),
            collate_fn=val_dataset.collate_fn if hasattr(val_dataset, "collate_fn") else None,
        )

        if "test_dataset" in data and isinstance(data["test_dataset"], Dataset):
            test_dataset = data.pop("test_dataset")
            self.test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.mini_batch_size,
                shuffle=False,
                num_workers=self.cfg["data_loader"].get("num_workers", 4),
                collate_fn=test_dataset.collate_fn if hasattr(test_dataset, "collate_fn") else None,
            )

        if data:
            for key, val in data.items():
                self.cfg["data"][key] = val

    @abstractmethod
    def _configure_optimizers(self):
        """Configure the training optimizer"""
        pass

    @abstractmethod
    def _configure_schedulers(self):
        """Configure the learning rate scheduler"""
        pass

    @abstractmethod
    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int) -> dict[str, float]:
        """Train a single epoch"""
        pass

    @abstractmethod
    def validate(self) -> dict[str, float]:
        """Validate after a single train epoch"""
        pass

    @abstractmethod
    def eval(self, *args, **kwargs) -> None:
        """
        Evaluate the model of the algorithm

        The parameters change according to the specific implementation logic of the subclass
        """
        pass

    def save(self, epoch: int, val_info: dict, best_loss: float, save_path: str) -> None:
        """Save checkpoint"""
        state = {
            "epoch": epoch,
            "cfg": self.cfg,
            "best loss": best_loss,
            "models": {},
            "optimizers": {},
        }

        for key, val in val_info.items():
            state.update({key: val})

        # save the models' parameters
        for key, val in self.models.items():
            state["models"].update({key: val.state_dict()})

        # save the optimizers' parameters
        for key, val in self._optimizers.items():
            state["optimizers"].update({key: val.state_dict()})

        # save the schedulers' parameters
        if hasattr(self, "schedulers"):
            state["schedulers"] = {k: v.state_dict() for k, v in self.schedulers.items()}

        torch.save(state, save_path)
        print(f"Saved checkpoint to {save_path}")

    def load(self, checkpoint: str) -> dict:
        state = torch.load(checkpoint, weights_only=False)

        # cfg
        self._cfg = state["cfg"]

        # load the models' parameters
        for key, val in self.models.items():
            val.load_state_dict(state["models"][key])

        # load the optimizers' parameters
        for key, val in self._optimizers.items():
            val.load_state_dict(state["optimizers"][key])

        # load the schedulers' parameters
        if hasattr(self, "schedulers"):
            for key, scheduler in self.schedulers.items():
                if key in state.get("schedulers", {}):
                    scheduler.load_state_dict(state["schedulers"][key])

        return state

    def set_train(self) -> None:
        """Set the pattern of all the models in the algorithm to training"""
        for model in self.models.values():
            model.train()
        self.training = True

    def set_eval(self) -> None:
        """Set the pattern of all the models in the algorithm to eval"""
        for model in self.models.values():
            model.eval()
        self.training = False
