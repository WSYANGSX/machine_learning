from typing import Literal, Mapping, Any, Union

import os
import yaml
from tqdm import tqdm
from abc import ABC, abstractmethod
from numbers import Integral, Real

import torch
from torch.amp import GradScaler
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from machine_learning.networks import BaseNet, NET_MAPS
from machine_learning.utils.logger import LOGGER
from machine_learning.types.aliases import FilePath
from machine_learning.utils import get_gpu_mem


class AlgorithmBase(ABC):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        data: Mapping[str, Union[Dataset, Any]],
        net: BaseNet | None = None,
        name: str | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
        amp: bool = False,
    ):
        """Base class of all algorithms

        Args:
            cfg (YamlFilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg dict.
            net (BaseNet): Neural neural required by the algorithm, provided by other algorithms externally.
            name (str, optional): Name of the algorithm. Defaults to None.
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
            data (Mapping[str, Union[Dataset, Any]]): Parsed specific dataset data, must including train dataset and val
            dataset, may contain data information of the specific dataset.
            amp (bool): Whether to enable Automatic Mixed Precision. Defaults to False.
        """
        super().__init__()

        LOGGER.info("Algorithm initializing by self...")

        # buffers
        self._nets = {}  # nets buffer
        self._optimizers = {}  # optimizers buffer
        self._schedulers = {}  # schedulers buffer

        # cfg
        self._cfg = self._load_config(cfg)
        self._validate_config()

        # name
        self._name = name if name is not None else self._cfg.get("algorithm", {}).get("name", __class__.__name__)

        # device
        self._device = self._configure_device(device)

        # settings
        self.training = False  # training mode or not
        self.last_opt_step = -1
        self.nbs = self.cfg["algorithm"].get("nbs", -1)
        self.batch_size = self.cfg["data_loader"].get("batch_size", 256)
        self.accumulate = max(1, round(self.nbs / self.batch_size))

        # build nets
        self._build_net(net)

        # initialize
        self._init_nets()
        self._init_amp(amp)
        self._init_dependent_on_data(data)

    @property
    def train_batches(self) -> int:
        return self._train_batches

    @property
    def val_batches(self) -> int:
        return self._val_batches

    @property
    def test_batches(self) -> int | None:
        return self._test_batches

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

    def _build_net(self, net: BaseNet) -> None:
        LOGGER.info(f"Building network of {self.name}...")

        self.net = net

        if self.net is not None:
            self._add_net("net", self.net)
        else:
            if issubclass(NET_MAPS[self.name], BaseNet):
                args = self.cfg["net"].get("args", {})
                if args is None:
                    raise ValueError("When a net is not created from the outside, the net parameters must be provided.")
                self.net = NET_MAPS[self.name](**args)
                self._add_net("net", self.net)

    def _init_on_trainer(self, cfg: dict[Any]) -> None:
        """Initialize the optimizers, and schedulers."""
        self._add_cfg("train", cfg)
        self._init_optimizers()
        self._init_schedulers()

    def _add_cfg(self, name, cfg: Any) -> None:
        # add additional configuration parameters may used by algo, e.g. traincfg
        self._cfg.update({name: cfg})

    def _add_net(self, name: str, net: BaseNet) -> None:
        LOGGER.info(f"Adding network {net.__class__.__name__}...")
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

    def _init_nets(self):
        """Configure nets of the algo, initialize the weight parameters of nets and show their structures"""
        LOGGER.info(f"Initializing the networks of {self.name}...")

        for net in self.nets.values():
            net.to(self._device)
            if self.cfg["net"]["initialize_weights"]:
                net._initialize_weights()
            net.view_structure()

    def _init_amp(self, amp: bool) -> None:
        LOGGER.info(f"Initializing the amp of {self.name}...")
        self.amp = amp
        self.scaler = GradScaler() if self.amp else None
        LOGGER.info(f"Automatic Mixed Precision (AMP) mode is {self.amp}.")

    def _init_dependent_on_data(self, data: Mapping[str, Union[Dataset, Any]]) -> None:
        """Initialize the training、validation(test) data loaders and other dataset-specific parameters.

        Args:
            data (Mapping[str, Union[Dataset, Any]]): Parsed specific dataset data, must including train dataset and val
            dataset, may contain data information of the specific dataset.
        """
        LOGGER.info(f"Initializing the data of {self.name}...")

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
        self._train_batches = len(self.train_loader)

        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg["data_loader"].get("num_workers", 4),
            collate_fn=val_dataset.collate_fn if hasattr(val_dataset, "collate_fn") else None,
        )
        self._val_batches = len(self.val_loader)

        if "test_dataset" in data and isinstance(data["test_dataset"], Dataset):
            test_dataset = data.pop("test_dataset")
            self.test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.cfg["data_loader"].get("num_workers", 4),
                collate_fn=test_dataset.collate_fn if hasattr(test_dataset, "collate_fn") else None,
            )
            self._test_batches = len(self.test_loader)
        else:
            self.test_loader = None
            self._test_batches = None

        if data:
            for key, val in data.items():
                self.cfg["data"][key] = val

    def _init_optimizers(self):
        """Configure the training optimizer"""
        LOGGER.info(f"Initializing the optimizers of {self.name}...")

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

        elif self.opt_cfg["type"] == "Adam":
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

    def _init_schedulers(self):
        """Configure the learning rate scheduler"""
        LOGGER.info(f"Initializing the schedulers of {self.name}...")

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

    def pbar_log(
        self,
        mode: Literal["train", "val"],
        pbar: tqdm,
        epoch: int | None = None,
        **kwargs,
    ) -> None:
        args = []
        format_str = "%-12s" * 2

        if mode == "train":
            epoch_str = "%g/%g" % (epoch + 1, self.cfg["train"]["epochs"])
            mem = "%.3gG" % get_gpu_mem()
            args.extend([epoch_str, mem])
        else:
            args.extend(["", ""])

        for _, val in kwargs.items():
            if isinstance(val, Integral):
                format_str += "%-12d"
                args.append(val)
            elif isinstance(val, Real):
                format_str += "%-12.4g"
                args.append(val)
            elif isinstance(val, str):
                format_str += "%-12s"
                args.append(val)
            elif val is None:
                format_str += "%-12s"
                args.append("")
            else:
                format_str += "%-12s"
                args.append(str(val))

        s = format_str % tuple(args)
        pbar.set_description(s)

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

    def print_metric_titles(self, mode: Literal["train", "val"], metrics: dict[str, Any]):
        if mode == "train":
            print(("\n" + "%-12s" * (len(metrics) + 2)) % ("Epoch", "gpu_mem", *metrics.keys()))
        elif mode == "val":
            print(("%-12s" * (len(metrics) + 2)) % ("", "", *metrics.keys()))

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int) -> dict[str, float]:
        """Train a single epoch"""
        self.set_train()

    def backward(self, loss: torch.Tensor) -> None:
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(self, batch_inters: int) -> None:
        if batch_inters - self.last_opt_step >= self.accumulate:
            if self.scaler:
                self.scaler.unscale_(
                    self.optimizer
                )  # unscale gradients, necessary if gradient clipping is performed after.
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg["optimizer"]["grad_clip"])

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.last_opt_step = batch_inters

    def validate(self) -> dict[str, float]:
        """Validate after a single train epoch"""
        self.set_eval()

    @abstractmethod
    def eval(self, *args, **kwargs) -> None:
        """
        Evaluate the model of the algorithm

        The parameters change according to the specific implementation logic of the subclass
        """
        pass

    def save(self, epoch: int, val_info: dict, best_loss: float, save_path: str, ckpt_dir: str) -> None:
        """Save checkpoint"""
        state = {
            "epoch": epoch,
            "cfg": self.cfg,
            "best loss": best_loss,
            "nets": {},
            "optimizers": {},
            "last_opt_step": self.last_opt_step,
            "amp": self.amp,
            "ckpt_dir": ckpt_dir,
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

        # save the scaler states
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()

        torch.save(state, save_path)
        LOGGER.info(f"Saved checkpoint to {save_path}")

    def load(self, checkpoint: str) -> dict:
        state = torch.load(checkpoint, weights_only=False)

        # cfg
        self._cfg = state["cfg"]
        self.last_opt_step = state["last_opt_step"]
        self.amp = state["amp"]

        # load the nets' parameters
        for key, val in self.nets.items():
            val.load_state_dict(state["nets"][key])
            val.to(self.device)

        # load the optimizers' parameters
        for key, val in self.optimizers.items():
            val.load_state_dict(state["optimizers"][key])

        # load the schedulers' parameters
        if hasattr(self, "schedulers"):
            for key, scheduler in self.schedulers.items():
                if key in state.get("schedulers", {}):
                    scheduler.load_state_dict(state["schedulers"][key])

        if self.amp:
            if self.scaler is not None:
                self.scaler.load_state_dict(state["scaler"])
            else:
                LOGGER.warning("Enable AMP to overwrite the current Settings based on checkpoint configuration.")
                self.scaler = GradScaler()
                self.scaler.load_state_dict(state["scaler"])
        else:
            if self.scaler is not None:
                LOGGER.warning(
                    "The checkpoint does not require AMP, overwrites its Settings and releases the current scaler."
                )
                self.scaler = None

        return state
