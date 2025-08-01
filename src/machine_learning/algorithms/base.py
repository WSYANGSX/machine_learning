from typing import Literal, Mapping, Any, Union

import os
import yaml
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter

from machine_learning.networks import BaseNet
from machine_learning.types.aliases import FilePath
from machine_learning.utils.logger import LOGGER


class AlgorithmBase(ABC):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        net: BaseNet | None = None,
        name: str | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ):
        """Base class of all algorithms

        Args:
            cfg (YamlFilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg dict.
            net (Mapping[str, BaseNet]): Neural neural required by the algorithm.
            name (str, optional): Name of the algorithm. Defaults to None.
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
        """
        super().__init__()

        self._nets = {}
        self._optimizers = {}
        self._schedulers = {}

        self.net = net
        if self.net:
            self._add_net("net", self.net)

        self.training = False

        # ------------------------ configure device -----------------------
        self._device = self._configure_device(device)

        # -------------------- load algo configuration --------------------
        self._cfg = self._load_config(cfg)
        self._validate_config()

        self.batch_size = self.cfg["data_loader"].get("batch_size", 256)
        self.nbs = self.cfg["algorithm"].get("nbs", -1)
        self.accumulate = max(1, round(self.nbs / self.batch_size))
        self.last_opt_step = -1

        # ---------------------- configure algo name ----------------------
        self._name = name if name is not None else self._cfg.get("algorithm", {}).get("name", __class__.__name__)

    @property
    def name(self) -> str:
        return self._name

    @property
    def nets(self) -> dict[str, BaseNet]:
        return self._nets

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

    def _initialize(self, data: Mapping[str, Union[Dataset, Any]]) -> None:
        """initialization algorithm, needs to be called before the training starts."""

        LOGGER.info("Algorithm initializing...")

        # --------------------- configure nets of algo --------------------
        self._configure_nets()

        # --------------------- configure optimizers ----------------------
        self._configure_optimizers()

        # --------------------- configure schedulers ----------------------
        self._configure_schedulers()

        # ------------------------ configure data -------------------------
        self._initialize_dependent_on_data(data)

    def _add_cfg(self, name, cfg: Any) -> None:
        # add additional configuration parameters may used by algo, e.g. traincfg
        self._cfg.update({name: cfg})

    def _add_net(self, name: str, net: BaseNet) -> None:
        self._nets.update({name: net})

    def _add_optimizer(self, name: str, optimizer: Optimizer) -> None:
        self._optimizers.update({name: optimizer})

    def _add_scheduler(self, name: str, scheduler: LRScheduler) -> None:
        self._schedulers.update({name: scheduler})

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

        return cfg

    def _validate_config(self):
        """Validate the config of the algorithm"""
        required_sections = ["algorithm", "net", "optimizer", "scheduler", "data_loader"]
        for section in required_sections:
            if section not in self.cfg:
                raise ValueError(f"The necessary parts are missing in the configuration file: {section}.")

    def _configure_nets(self):
        """Configure nets of the algo, initialize the weight parameters of nets and show their structures"""
        for net in self.nets.values():
            net.to(self._device)
            if self.cfg["net"]["initialize_weights"]:
                net._initialize_weights()
            net.view_structure()

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
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg["data_loader"].get("num_workers", 4),
            collate_fn=val_dataset.collate_fn if hasattr(val_dataset, "collate_fn") else None,
        )

        if "test_dataset" in data and isinstance(data["test_dataset"], Dataset):
            test_dataset = data.pop("test_dataset")
            self.test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.cfg["data_loader"].get("num_workers", 4),
                collate_fn=test_dataset.collate_fn if hasattr(test_dataset, "collate_fn") else None,
            )

        if data:
            for key, val in data.items():
                self.cfg["data"][key] = val

        self.num_batches = len(self.train_loader)

    def _configure_optimizers(self):
        """Configure the training optimizer"""
        self.opt_cfg = self._cfg["optimizer"]

        self.optimizer = None

        if self.opt_cfg["type"] == "SGD":
            self.optimizer = torch.optim.SGD(
                params=self.net.parameters(),
                lr=self.opt_cfg["lr"],
                momentum=self.opt_cfg["momentum"],
                weight_decay=self.opt_cfg["weight_decay"],
            )
            self._add_optimizer("optimizer", self.optimizer)

        if self.opt_cfg["type"] == "Adam":
            self.optimizer = torch.optim.Adam(
                params=self.net.parameters(),
                lr=self.opt_cfg["lr"],
                betas=(self.opt_cfg["beta1"], self.opt_cfg["beta2"]),
                eps=self.opt_cfg["eps"],
                weight_decay=self.opt_cfg["weight_decay"],
            )
            self._add_optimizer("optimizer", self.optimizer)

        else:
            ValueError(f"Does not support optimizer:{self.opt_cfg['type']} currently.")

    def _configure_schedulers(self):
        """Configure the learning rate scheduler"""
        self.sch_config = self._cfg["scheduler"]

        self.scheduler = None

        if self.sch_config.get("type") == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.sch_config.get("factor", 0.1),
                patience=self.sch_config.get("patience", 10),
            )
            self._add_scheduler("scheduler", self.scheduler)

    def __str__(self) -> str:
        """Return a summary string of the algorithm instance."""
        # Basic info
        summary = [
            f"Algorithm: {self.name}",
            f"Device: {self.device}",
            f"Batch size: {self.batch_size}",
        ]

        # Networks info
        nets_info = []
        for name, net in self.nets.items():
            num_params = sum(p.numel() for p in net.parameters())
            nets_info.append(f"  {name}: {type(net).__name__} ({num_params / 1e6:.2f}M params)")
        summary.append("Networks:\n" + "\n".join(nets_info) if nets_info else "Networks: None")

        # Optimizers info
        optim_info = []
        for name, opt in self.optimizers.items():
            lr = opt.param_groups[0]["lr"] if opt.param_groups else "N/A"
            optim_info.append(f"  {name}: {type(opt).__name__} (lr={lr:.2e})")
        summary.append("Optimizers:\n" + "\n".join(optim_info) if optim_info else "Optimizers: None")

        # Schedulers info
        sched_info = []
        for name, sched in self.schedulers.items():
            sched_info.append(f"  {name}: {type(sched).__name__}")
        summary.append("Schedulers:\n" + "\n".join(sched_info) if sched_info else "Schedulers: None")

        return "\n".join(summary)

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
            "nets": {},
            "optimizers": {},
            "last_opt_step": self.last_opt_step,
        }

        for key, val in val_info.items():
            state.update({key: val})

        # save the nets' parameters
        for key, val in self.nets.items():
            state["nets"].update({key: val.state_dict()})

        # save the optimizers' parameters
        for key, val in self.optimizers.items():
            state["optimizers"].update({key: val.state_dict()})

        # save the schedulers' parameters
        if hasattr(self, "schedulers"):
            state["schedulers"] = {k: v.state_dict() for k, v in self.schedulers.items()}

        torch.save(state, save_path)
        LOGGER.info(f"Saved checkpoint to {save_path}")

    def load(self, checkpoint: str) -> dict:
        state = torch.load(checkpoint, weights_only=False)

        # cfg
        self._cfg = state["cfg"]
        self.last_opt_step = state["last_opt_step"]

        # load the nets' parameters
        for key, val in self.nets.items():
            val.load_state_dict(state["nets"][key])

        # load the optimizers' parameters
        for key, val in self.optimizers.items():
            val.load_state_dict(state["optimizers"][key])

        # load the schedulers' parameters
        if hasattr(self, "schedulers"):
            for key, scheduler in self.schedulers.items():
                if key in state.get("schedulers", {}):
                    scheduler.load_state_dict(state["schedulers"][key])

        return state

    def set_train(self) -> None:
        """Set the pattern of all the nets in the algorithm to training"""
        for net in self.nets.values():
            net.train()
        self.training = True

    def set_eval(self) -> None:
        """Set the pattern of all the nets in the algorithm to eval"""
        for net in self.nets.values():
            net.eval()
        self.training = False
