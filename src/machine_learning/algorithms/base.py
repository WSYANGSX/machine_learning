import os
import yaml
from typing import Literal, Mapping, Any
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from machine_learning.models import BaseNet
from machine_learning.types.aliases import FilePath
from machine_learning.utils.others import print_dict, print_segmentation


class AlgorithmBase(ABC):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        models: Mapping[str, BaseNet],
        name: str | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ):
        """Base class of all algorithms

        Args:
            cfg (YamlFilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg dict.
            models (Mapping[str, BaseNet]): Models required by the algorithm, {"net1": BaseNet1, "net2": BaseNet2}.
            name (str, optional): Name of the algorithm. Defaults to None.
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to "auto"-automatic selection by algorithm.
        """
        super().__init__()

        self._models = models
        self._optimizers = {}
        self._schedulers = {}

        # ------------------------ configure device -----------------------
        self._device = self._configure_device(device)

        # -------------------- load algo configuration --------------------
        self._cfg = self._load_config(cfg)
        self._validate_config()

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
        required_sections = ["algorithm", "model", "optimizer", "scheduler"]
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

    def _initialize_dependent_on_data(
        self,
        batch_size: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
        **kwargs,
    ) -> None:
        """Initialize the training and validation data loaders and other dataset-specific parameters, need to be called
        before training.

        Args:
            train_loader (DataLoader): train dataset loader.
            val_loader (DataLoader): val dataset loader.
        """
        self.batch_size = batch_size

        self.train_loader = train_loader
        self.val_loader = val_loader
        if test_loader:
            self.test_loader = test_loader

        # Unique parameters of the data set
        protected_attrs = {"train_loader", "val_loader", "batch_size", "test_loader"}

        if kwargs:
            self.cfg["data"] = {}

            for key, val in kwargs.items():
                if key in protected_attrs:
                    print(f"Attempted to override protected attribute '{key}'. Ignored.")
                    continue

                if hasattr(self, key):
                    setattr(self, key, val)
                    print(f"[INFO] Set {key} attribute of {self.__class__.__name__} to new value.")
                else:
                    setattr(self, key, val)
                    print(f"[INFO] {self.__class__.__name__.capitalize()} set new attribute: {key}")

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

    def save(self, epoch: int, loss: dict, best_loss: float, save_path: str) -> None:
        """Save checkpoint"""
        state = {"epoch": epoch, "cfg": self.cfg, "loss": loss, "best loss": best_loss, "models": {}, "optimizers": {}}

        # save the model parameters
        for key, val in self.models.items():
            state["models"].update({key: val.state_dict()})

        # save the optimizer parameters
        for key, val in self._optimizers.items():
            state["optimizers"].update({key: val.state_dict()})

        torch.save(state, save_path)
        print(f"Saved checkpoint to {save_path}")

    def load(self, checkpoint: str) -> tuple[Any]:
        state = torch.load(checkpoint)

        # load the model parameters
        for key, val in self.models.items():
            val.load_state_dict(state["models"][key])

        # load the optimizer parameters
        for key, val in self._optimizers.items():
            val.load_state_dict(state["optimizers"][key])

        epoch = state["epoch"]
        cfg = state["cfg"]
        loss = state["loss"]
        best_loss = state["best loss"]

        return {"epoch": epoch, "cfg": cfg, "loss": loss, "best_loss": best_loss}

    def set_train(self) -> None:
        """Set the pattern of all the models in the algorithm to training"""
        for model in self.models.values():
            model.train()

    def set_eval(self) -> None:
        """Set the pattern of all the models in the algorithm to eval"""
        for model in self.models.values():
            model.eval()
