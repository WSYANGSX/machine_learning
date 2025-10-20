from typing import Literal, Mapping, Any

import os
from tqdm import tqdm
from abc import ABC, abstractmethod
from numbers import Integral, Real

import torch
from torch.amp import GradScaler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter

from machine_learning.utils.logger import LOGGER
from machine_learning.networks import BaseNet, NET_MAPS
from machine_learning.utils import get_gpu_mem, load_cfg
from machine_learning.utils.constants import DATACFG_PATH, ALGOCFG_PATH
from machine_learning.dataset import ParserBase, PARSER_MAPS, build_dataset, build_dataloader


class AlgorithmBase(ABC):
    def __init__(
        self,
        cfg: str | Mapping[str, Any],
        net: BaseNet | None = None,
        name: str | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
        amp: bool | None = None,
    ):
        """Base class of all algorithms

        Args:
            cfg (FilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg dict.
            net (BaseNet): Neural neural required by the algorithm, provided by other algorithms externally.
            name (str, optional): Name of the algorithm. Defaults to None.
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
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
        self.nbs = self.cfg["data"]["nbs"]
        self.batch_size = self.cfg["data"]["batch_size"]
        self.accumulate = max(1, round(self.nbs / self.batch_size))

        # net
        self.provided_net = net

        # amp
        self._init_amp(amp)

    def _init_on_trainer(
        self,
        train_cfg: dict[str, Any],
        dataset: str | Mapping[str, Any],
    ) -> None:
        """Initialize the datasets, dataloaders, nets, optimizers, and schedulers. And the attributes that require the dataset_cfg and trainer_cfg are created here."""
        self._add_cfg("trainer", train_cfg)
        self._trainer_cfg = train_cfg

        # init train/val datasets and dataloaders
        self._init_train_datasets(dataset)
        self._init_train_dataloaders()

        # build net
        self._build_net(self.provided_net)
        self._init_nets()

        # init opts and schedulers
        self._init_optimizers()
        self._init_schedulers()

    def _init_on_evaluator(self, ckpt: str, dataset: str | Mapping[str, Any]) -> None:
        # init test dataset and test dataloader
        self._init_eval_dataset(dataset)
        self._init_eval_dataloader()

        # build net
        self._build_net(self.provided_net)
        self.load(ckpt)

    @property
    def train_dataset(self) -> None | Dataset:
        return self._train_dataset if hasattr(self, "_train_dataset") else None

    @property
    def val_dataset(self) -> None | Dataset:
        return self._val_dataset if hasattr(self, "_val_dataset") else None

    @property
    def test_dataset(self) -> None | Dataset:
        return self._test_dataset if hasattr(self, "_test_dataset") else None

    @property
    def train_loader(self) -> None | DataLoader:
        return self._train_loader if hasattr(self, "_train_loader") else None

    @property
    def val_loader(self) -> None | DataLoader:
        return self._val_loader if hasattr(self, "_val_loader") else None

    @property
    def test_loader(self) -> None | DataLoader:
        return self._test_loader if hasattr(self, "_test_loader") else None

    @property
    def train_batches(self) -> int:
        return self._train_batches if hasattr(self, "_train_batches") else None

    @property
    def val_batches(self) -> int:
        return self._val_batches if hasattr(self, "_val_batches") else None

    @property
    def test_batches(self) -> int:
        return self._val_batches if hasattr(self, "_test_batches") else None

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
    def flatten_cfg(self) -> dict:
        return self._flatten_cfg

    @property
    def dataset_cfg(self) -> dict:
        return self._dataset_cfg

    @property
    def trainer_cfg(self) -> dict:
        return self._trainer_cfg

    @property
    def device(self) -> torch.device:
        return self._device

    def _load_config(self, cfg: str | Mapping[str, Any]) -> dict:
        if isinstance(cfg, str):
            cfg = os.path.join(ALGOCFG_PATH, cfg)
        cfg = load_cfg(cfg)

        return cfg

    def _load_datasetcfg(self, cfg: str | Mapping[str, Any]) -> dict:
        if isinstance(cfg, str):
            cfg = os.path.join(DATACFG_PATH, cfg)
        cfg = load_cfg(cfg)

        return cfg

    def _build_net(self, net: BaseNet) -> None:
        LOGGER.info(f"Building network of {self.name}...")

        if net is not None:
            self.net = net
            self._add_net("net", self.net)
        else:
            LOGGER.info("No outside net provided, building net from default configuration...")
            if issubclass(NET_MAPS[self.name], BaseNet):
                self.net = NET_MAPS[self.name](**self.flatten_cfg)
                self._add_net("net", self.net)

    def _add_cfg(self, name, cfg: dict[str, Any]) -> None:
        """add additional configuration parameters."""
        if name not in self.cfg:
            self.cfg[name] = cfg
        else:
            self.cfg[name].update(cfg)

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

    def _validate_config(self):
        """Validate the config of the algorithm"""
        required_sections = ["algorithm", "net", "optimizer", "scheduler", "data"]
        for section in required_sections:
            if section not in self.cfg:
                raise ValueError(f"The necessary parts are missing in the configuration file: {section}.")

    def parse_dataset(self, dataset: str | Mapping[str, Any]) -> tuple:
        """Parse dataset info by configuration."""
        LOGGER.info("Parsing dataset cfg...")
        dataset_cfg = self._load_datasetcfg(dataset)
        self._add_cfg("data", {"dataset": dataset_cfg})
        self._dataset_cfg = dataset_cfg

        # parser data
        dataset_name = dataset_cfg["name"]
        parser: ParserBase = PARSER_MAPS[dataset_name](dataset_cfg)
        parsing = parser.parse()
        trian_parsing, val_parsing, test_parsing = parsing["train"], parsing["val"], parsing.get("test", {})

        # build dataset
        type = self.dataset_cfg.get("dataset_type", None)
        if type is None:
            raise ValueError("The data type must be provided for Dataset mapping.")

        return type, trian_parsing, val_parsing, test_parsing

    def _init_train_datasets(self, dataset: str | Mapping[str, Any]) -> None:
        """Initialize train and val datasets of the algorithm."""
        type, trian_parsing, val_parsing, _ = self.parse_dataset(dataset)

        LOGGER.info("Getting train and val datasets...")
        self._flatten_cfg = self.cfg_flat(self.cfg)
        self._train_dataset = build_dataset(type, self.flatten_cfg, trian_parsing, self.batch_size, "train")
        self._val_dataset = build_dataset(type, self.flatten_cfg, val_parsing, self.batch_size, "val")

    def _init_eval_dataset(self, dataset: str | Mapping[str, Any]) -> None:
        """Initialize the test dataset of the algorithm."""
        type, _, val_parsing, test_parsing = self.parse_dataset(dataset)

        LOGGER.info("Getting the test dataset...")
        self._flatten_cfg = self.cfg_flat(self.cfg)
        if test_parsing:
            self._test_dataset = build_dataset(type, self.flatten_cfg, test_parsing, self.batch_size, "test")
        else:
            self._test_dataset = build_dataset(type, self.flatten_cfg, val_parsing, self.batch_size, "test")

    def _init_train_dataloaders(self) -> None:
        """Initialize train and val dataloaders."""
        LOGGER.info(f"Initializing train and val dataloaders of {self.name}...")

        self._train_loader = build_dataloader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            workers=self.cfg["data"].get("workers", 8),
            shuffle=self.cfg["data"].get("shuffle"),
            pin_memory=self.cfg["data"].get("pin_memory", False),
            mode="train",
        )
        self._train_batches = len(self.train_loader)

        self._val_loader = build_dataloader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            workers=self.cfg["data"].get("workers", 8),
            shuffle=False,
            pin_memory=self.cfg["data"].get("pin_memory", False),
            mode="train",
        )
        self._val_batches = len(self.val_loader)

    def _init_eval_dataloader(self) -> None:
        """Initialize the test dataloader."""
        LOGGER.info(f"Initializing the test dataloader of {self.name}...")

        self._test_loader = build_dataloader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            workers=self.cfg["data"].get("workers", 8),
            shuffle=False,
            pin_memory=self.cfg["data"].get("pin_memory", False),
            mode="test",
        )
        self._test_batches = len(self.test_loader)

    def _init_nets(self):
        """Configure nets of the algo, initialize the weight parameters of nets and show their structures"""
        LOGGER.info(f"Initializing the networks of {self.name}...")

        for net in self.nets.values():
            net.to(self._device)
            if self.cfg["net"]["initialize_weights"]:
                net._initialize_weights()
            net.view_structure()

    def _init_amp(self, amp: bool | None = None) -> None:
        self.amp = amp if amp is not None else self._cfg["algorithm"].get("amp", False)
        self.scaler = GradScaler() if self.amp else None
        LOGGER.info(f"Automatic Mixed Precision (AMP) mode is {self.amp}.")

    def _init_optimizers(self):
        """Configure the training optimizer"""
        LOGGER.info(f"Initializing the optimizers of {self.name}...")

        self.opt_cfg = self._cfg["optimizer"]

        self.optimizer = None

        if self.opt_cfg["opt"] == "SGD":
            self.optimizer = torch.optim.SGD(
                params=self.net.parameters(),
                lr=self.opt_cfg["lr"],
                momentum=self.opt_cfg["momentum"],
                weight_decay=self.opt_cfg["weight_decay"],
            )
            self._add_optimizer("optimizer", self.optimizer)

        elif self.opt_cfg["opt"] == "Adam":
            self.optimizer = torch.optim.Adam(
                params=self.net.parameters(),
                lr=self.opt_cfg["lr"],
                betas=(self.opt_cfg["beta1"], self.opt_cfg["beta2"]),
                eps=self.opt_cfg["eps"],
                weight_decay=self.opt_cfg["weight_decay"],
            )
            self._add_optimizer("optimizer", self.optimizer)

        else:
            ValueError(f"Does not support optimizer:{self.opt_cfg['opt']} currently.")

    def _init_schedulers(self):
        """Initialize the learning rate scheduler"""
        LOGGER.info(f"Initializing the schedulers of {self.name}...")

        self.sch_config = self._cfg["scheduler"]

        self.scheduler = None

        if self.sch_config.get("sched") == "ReduceLROnPlateau":
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

    def cfg_flat(self, cfg: dict[str, Any]) -> dict[str, Any]:
        items = []
        for key, value in cfg.items():
            if isinstance(value, dict):
                items.extend(self.cfg_flat(value).items())
            else:
                items.append((key, value))

        return dict(items)

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
            epoch_str = "%g/%g" % (epoch + 1, self.cfg["trainer"]["epochs"])
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
        self._cfg = state["cfg"]  # include dataset, trainer
        self._dataset_cfg = self._cfg["data"].get("dataset", {})
        self._trainer_cfg = self._cfg["data"].get("trainer", {})

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
