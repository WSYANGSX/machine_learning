from typing import Literal, Mapping, Any

import os
from tqdm import tqdm
from numbers import Integral, Real
from abc import ABC, abstractmethod

import torch
from torch.amp import autocast, GradScaler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter

from machine_learning.utils.logger import LOGGER
from machine_learning.utils.streams import StreamBase
from machine_learning.networks import BaseNet, NET_MAPS
from machine_learning.utils.torch_utils import ModelEMA
from machine_learning.utils import get_gpu_mem, load_cfg
from machine_learning.utils.constants import DATACFG_PATH, ALGOCFG_PATH
from machine_learning.data.dataset import ParserBase, PARSER_MAPS, build_dataset, build_dataloader


class AlgorithmBase(ABC):
    def __init__(
        self,
        cfg: str | Mapping[str, Any],
        net: BaseNet | None = None,
        name: str | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
        amp: bool | None = None,
        ema: bool | None = False,
    ):
        """Base class of all algorithms.

        Args:
            cfg (FilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg dict.
            net (BaseNet): Neural neural required by the algorithm, provided by other algorithms externally.
            name (str, optional): Name of the algorithm. Defaults to None.
            device (Literal["cuda", "cpu", "auto"], optional): Running device. Defaults to
            "auto"-automatic selection by algorithm.
            amp (bool): Whether to enable Automatic Mixed Precision. Defaults to False.
            ema (bool): Whether to enable Exponential Moving Average. Defaults to False.
        """
        super().__init__()

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

        self.provided_net = net  # net
        self.ema_enable = ema if ema is not None else self._cfg["algorithm"].get("ema", False)  # ema
        self.amp_enable = amp if amp is not None else self._cfg["algorithm"].get("amp", False)  # amp

    def _init_on_trainer(
        self,
        train_cfg: dict[str, Any],
        dataset: str | Mapping[str, Any],
    ) -> None:
        """
        Initialize the datasets, dataloaders, nets, optimizers, and schedulers. And the attributes that require the
        dataset cfg and trainer cfg are created here.

        Args:
            train_cfg (Mapping[str, Any]): Configuration of the trainer.
            dataset (FilePath | Mapping[str, Any]): Configuration of the dataset, it can be yaml file path or cfg dict.
        """
        self._add_cfg("trainer", train_cfg)
        self._trainer_cfg = train_cfg
        self.epochs = train_cfg.get("epochs")

        # init train/val datasets and dataloaders
        self._init_train_datasets(dataset)
        self._init_train_dataloaders()

        # build net
        self._build_net(self.provided_net)
        self._init_nets()

        # init ema
        if self.ema_enable:
            LOGGER.info("EMA is enabled.")
            self._init_ema()

        # init amp
        self._init_amp()

        # init opts and schedulers
        self._init_optimizers()
        self._init_schedulers()

    def _init_on_evaluator(
        self,
        ckpt: str,
        dataset: str | Mapping[str, Any] | None = None,
        load_dataset: bool = True,
        plot: bool | None = False,
        save_dir: str | None = None,
    ) -> None:
        """
        Initialize the evaluation dataset, dataloader, net, and load the checkpoint weights.

        Args:
            ckpt (FilePath): Checkpoint file path.
            dataset (FilePath | Mapping[str, Any]): Configuration of the dataset, it can be yaml file path or cfg dict.
            load_dataset (bool): Whether to use the dataset provided by the evaluator.
        """
        self.plot = plot
        self.save_dir = save_dir if save_dir is not None else "./"

        self.ema_enable = False  # disable ema for evaluation
        self.amp_enable = False
        # init test dataset and test dataloader
        if load_dataset and dataset is not None:
            self._init_eval_dataset(dataset)
            self._init_eval_dataloader()
        else:
            self.parse_dataset(dataset)  # add dataset_cfg
            self._flatten_cfg = self.cfg_flat(self.cfg)  # used for building net

        # build net
        self._build_net(self.provided_net)
        # load net weights from ema
        self.load(ckpt, load_ema=True)
        # set device
        for net in self.nets.values():
            net.to(self.device)

        self.set_eval()

    def _init_on_predictor(
        self,
        ckpt: str,
        dataset: str | Mapping[str, Any] | None = None,
    ) -> None:
        """
        Initialize the net, and load the checkpoint weights.

        Args:
            ckpt (FilePath): Checkpoint file path.
        """
        self._init_on_evaluator(ckpt=ckpt, dataset=dataset, load_dataset=False)

    @property
    def train_dataset(self) -> None | Dataset:
        """Return the training dataset if initialized."""
        return self._train_dataset if hasattr(self, "_train_dataset") else None

    @property
    def val_dataset(self) -> None | Dataset:
        """Return the validation dataset if initialized."""
        return self._val_dataset if hasattr(self, "_val_dataset") else None

    @property
    def test_dataset(self) -> None | Dataset:
        """Return the test dataset if initialized."""
        return self._test_dataset if hasattr(self, "_test_dataset") else None

    @property
    def train_loader(self) -> None | DataLoader:
        """Return the training dataloader if initialized."""
        return self._train_loader if hasattr(self, "_train_loader") else None

    @property
    def val_loader(self) -> None | DataLoader:
        """Return the validation dataloader if initialized."""
        return self._val_loader if hasattr(self, "_val_loader") else None

    @property
    def test_loader(self) -> None | DataLoader:
        """Return the test dataloader if initialized."""
        return self._test_loader if hasattr(self, "_test_loader") else None

    @property
    def train_batches(self) -> int | None:
        """Return the number of training batches if initialized."""
        return self._train_batches if hasattr(self, "_train_batches") else None

    @property
    def val_batches(self) -> int | None:
        """Return the number of validation batches if initialized."""
        return self._val_batches if hasattr(self, "_val_batches") else None

    @property
    def test_batches(self) -> int | None:
        """Return the number of test batches if initialized."""
        return self._test_batches if hasattr(self, "_test_batches") else None

    @property
    def name(self) -> str:
        """Return the name of the algorithm."""
        return self._name

    @property
    def nets(self) -> dict[str, BaseNet]:
        """Return the networks of the algorithm."""
        return self._nets

    @property
    def optimizers(self) -> dict[str, Optimizer]:
        """Return the optimizers of the algorithm."""
        return self._optimizers

    @property
    def schedulers(self) -> dict[str, LRScheduler]:
        """Return the schedulers of the algorithm."""
        return self._schedulers

    @property
    def cfg(self) -> dict:
        """Return the configuration of the algorithm."""
        return self._cfg

    @property
    def flatten_cfg(self) -> dict:
        """Return the flattened configuration of the algorithm."""
        return self._flatten_cfg

    @property
    def dataset_cfg(self) -> dict:
        """Return the dataset configuration of the algorithm."""
        return self._dataset_cfg

    @property
    def trainer_cfg(self) -> dict:
        """Return the trainer configuration of the algorithm."""
        return self._trainer_cfg

    @property
    def device(self) -> torch.device:
        """Return the running device of the algorithm."""
        return self._device

    def _load_config(self, cfg: str | Mapping[str, Any]) -> dict:
        """
        Load configuration from file path or dict.

        Args:
            cfg (FilePath | Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg dict.
        """
        if isinstance(cfg, str):
            cfg = os.path.join(ALGOCFG_PATH, cfg)
        cfg = load_cfg(cfg)

        return cfg

    def _load_datasetcfg(self, cfg: str | Mapping[str, Any]) -> dict:
        """
        Load dataset configuration from file path or dict.

        Args:
            cfg (FilePath | Mapping[str, Any]): Dataset configuration of the algorithm (yaml file path or cfg dict).
        """
        if isinstance(cfg, str):
            cfg = os.path.join(DATACFG_PATH, cfg)
        cfg = load_cfg(cfg)

        return cfg

    def _build_net(self, net: BaseNet) -> None:
        """
        Build the network of the algorithm.

        Args:
            net (BaseNet): Neural neural required by the algorithm, provided externally.
        """
        LOGGER.info(f"Building network of {self.name}...")

        if net is not None:
            self.net = net
            self._add_net("net", self.net)
            # override parameters of net
            for key in self.cfg["net"]:
                if key in self.net.__dict__:
                    self.cfg["net"][key] = self.net.__dict__[key]
        else:
            LOGGER.info(f"No outside nets provided, building nets of {self.name} from default configuration...")
            if issubclass(NET_MAPS[self.name], BaseNet):
                self.net = NET_MAPS[self.name](**self.flatten_cfg)
                self._add_net("net", self.net)

        # add net name to cfg
        self._add_cfg("net", {"net_name": self.net.__class__.__name__})

    def _add_cfg(self, name, cfg: dict[str, Any]) -> None:
        """
        Add additional configuration parameters.

        Args:
            name (str): The name of the configuration section.
            cfg (Mapping[str, Any]): The configuration dictionary to add.
        """
        if name not in self.cfg:
            self.cfg[name] = cfg
        else:
            self.cfg[name].update(cfg)

    def _add_net(self, name: str, net: BaseNet) -> None:
        """
        Add a network to the algorithm.

        Args:
            name (str): The name of the network.
            net (BaseNet): The network instance to add.
        """
        LOGGER.info(f"Adding network {net.__class__.__name__}...")
        self._nets.update({name: net})

    def _add_optimizer(self, name: str, optimizer: Optimizer) -> None:
        """
        Add an optimizer to the algorithm.

        Args:
            name (str): The name of the optimizer.
            optimizer (Optimizer): The optimizer instance to add.
        """
        self._optimizers.update({name: optimizer})

    def _add_scheduler(self, name: str, scheduler: LRScheduler) -> None:
        """
        Add a scheduler to the algorithm.

        Args:
            name (str): The name of the scheduler.
            scheduler (LRScheduler): The scheduler instance to add.
        """
        self._schedulers.update({name: scheduler})

    def _configure_device(self, device: str) -> torch.device:
        """
        Configure the running device of the algorithm.

        Args:
            device (str): The device to run the algorithm on.
        """
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _validate_config(self):
        """
        Validate the necessary config of the algorithm.
        """
        required_sections = ["algorithm", "net", "optimizer", "scheduler", "data"]
        for section in required_sections:
            if section not in self.cfg:
                raise ValueError(f"The necessary parts are missing in the configuration file: {section}.")

    def parse_dataset(self, dataset: str | Mapping[str, Any]) -> tuple:
        """
        Parse dataset info by configuration.

        Args:
            dataset (str | Mapping[str, Any]): The dataset configuration.
        """
        LOGGER.info("Parsing dataset cfg...")
        self._dataset_cfg = self._load_datasetcfg(dataset)
        self._add_cfg("data", {"dataset": self.dataset_cfg})

        # parser data
        dataset_name = self._dataset_cfg["name"]
        parser: ParserBase = PARSER_MAPS[dataset_name](self._dataset_cfg)
        parsing = parser.parse()
        train_parsing, val_parsing, test_parsing = parsing["train"], parsing["val"], parsing.get("test", {})

        # build dataset
        type = self.dataset_cfg.get("dataset_type", None)
        if type is None:
            raise ValueError("The data type must be provided for Dataset mapping.")

        return type, train_parsing, val_parsing, test_parsing

    def _init_ema(self) -> None:
        """
        Initialize Exponential Moving Average (EMA) for the networks.
        """
        LOGGER.info(f"Initializing EMA for the networks of {self.name}...")
        self.emas = {}

        for key, net in self.nets.items():
            self.emas[key] = ModelEMA(net)

    def _init_amp(self) -> None:
        if self.amp_enable:
            LOGGER.info("Automatic Mixed Precision (AMP) enabled.")
            self.scaler = GradScaler()
        else:
            LOGGER.info("Automatic Mixed Precision (AMP) disenabled.")
            self.scaler = None

    def _init_train_datasets(self, dataset: str | Mapping[str, Any]) -> None:
        """Initialize train and val datasets of the algorithm."""
        type, train_parsing, val_parsing, _ = self.parse_dataset(dataset)

        LOGGER.info("Getting train and val datasets...")
        self._flatten_cfg = self.cfg_flat(self.cfg)
        self._train_dataset = build_dataset(type, self.flatten_cfg, train_parsing, self.batch_size, "train")
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
            mode="val",
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

    def _init_optimizers(self):
        """Configure the training optimizer"""
        LOGGER.info(f"Initializing the optimizers of {self.name}...")

        self.optimizer = None
        opt_cfg = self._cfg["optimizer"]
        opt_type = opt_cfg.get("opt_type", None)
        if opt_type is None:
            raise ValueError("Please provide opt_type parameter in algorithm cfg file under optimizer item.")

        if opt_type == "SGD":
            self.optimizer = torch.optim.SGD(
                params=self.net.parameters(),
                lr=opt_cfg["lr"],
                momentum=opt_cfg["momentum"],
                weight_decay=opt_cfg["weight_decay"],
            )
            self._add_optimizer("optimizer", self.optimizer)

        elif opt_type == "Adam":
            self.optimizer = torch.optim.Adam(
                params=self.net.parameters(),
                lr=opt_cfg["lr"],
                betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
                eps=opt_cfg["eps"],
                weight_decay=opt_cfg["weight_decay"],
            )
            self._add_optimizer("optimizer", self.optimizer)

        else:
            raise ValueError(f"Does not support optimizer:{opt_type} currently.")

    def _init_schedulers(self):
        """Initialize the learning rate scheduler"""
        LOGGER.info(f"Initializing the schedulers of {self.name}...")

        self.scheduler = None
        sch_cfg = self._cfg["scheduler"]
        sch_type = sch_cfg.get("sch_type", None)

        if sch_type == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=sch_cfg.get("mode", "min"),
                factor=sch_cfg.get("factor", 0.1),
                patience=sch_cfg.get("patience", 10),
            )
            self._add_scheduler("scheduler", self.scheduler)

        else:
            raise ValueError(f"Does not support scheduler:{sch_type} currently.")

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
        mode: Literal["train", "val", "test"],
        pbar: tqdm,
        epoch: int | None = None,
        **kwargs,
    ) -> None:
        args = []
        format_str = "%-14s" * 2 if mode in ("train", "val") else ""

        if mode == "train":
            epoch_str = "%g/%g" % (epoch + 1, self.cfg["trainer"]["epochs"])
            mem = "%.3gG" % get_gpu_mem()
            args.extend([epoch_str, mem])
        elif mode == "val":
            args.extend(["", ""])

        for _, val in kwargs.items():
            if isinstance(val, Integral):
                format_str += "%-14d"
                args.append(val)
            elif isinstance(val, Real):
                format_str += "%-14.4g"
                args.append(val)
            elif isinstance(val, str):
                format_str += "%-14s"
                args.append(val)
            elif val is None:
                format_str += "%-14s"
                args.append("")
            else:
                format_str += "%-14s"
                args.append(str(val))

        s = format_str % tuple(args)
        pbar.set_description(s)

    def print_metric_titles(self, mode: Literal["train", "val", "test"], metrics: dict[str, Any]):
        if mode == "train":
            print(("\n" + "%-14s" * (len(metrics) + 2)) % ("Epoch", "gpu_mem", *metrics.keys()))
        elif mode == "val":
            print(("%-14s" * (len(metrics) + 2)) % ("", "", *metrics.keys()))
        else:
            print(("%-14s" * len(metrics)) % tuple(metrics.keys()))

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

    def train_epoch(
        self, epoch: int, writer: SummaryWriter, log_interval: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Returns training metrics and info dict for the epoch."""
        self.set_train()
        self.on_epoch_start(epoch)

        info = {}  # Record the information of each epoch.
        statistics = []  # Record the process data of each batch for the calculation of the final indicators.
        metrics = self._init_metrics("train")
        self.print_metric_titles("train", metrics)

        pbar = tqdm(enumerate(self.train_loader), total=self.train_batches)
        for i, batch in pbar:
            # Warmup
            batches = epoch * self.train_batches + i
            self.warmup(batches, epoch)

            data = self._prepare_batch(batch, "train")

            # Loss calculation
            with autocast(device_type=self.device.type, enabled=self.amp_enable):  # covers the forward computation
                res = self._forward_batch(self.net, data, "train")
                loss = res["loss"]

            # Gradient backpropagation
            self.backward(loss)
            # Parameter optimization
            self.optimizer_step(batches)

            # Metrics
            self._post_process(i, res, batch, data, metrics, statistics, info, "train")

            if i % log_interval == 0:
                self.write(batches, writer, res, metrics)

            # log
            self.pbar_log("train", pbar, epoch, **metrics)

        return metrics, info

    def on_epoch_start(self, epoch: int) -> None:
        """Lifecycle hook. etc. close Mosaic."""
        pass

    def warmup(self, batches: int, epoch: int) -> None:
        """Learning rate warm up."""
        pass

    @abstractmethod
    def _init_metrics(self, mode: Literal["train", "val", "test"]) -> dict[str, Any]:
        """Metrics used for different mode to print information to console."""
        raise NotImplementedError("_init_metrics method must be implemented.")

    @abstractmethod
    def _prepare_batch(self, batch: dict[str, Any], mode: Literal["train", "val", "test"]) -> dict[str, Any]:
        """Prepare different batch data for different modes."""
        raise NotImplementedError("_prepare_batch method must be implemented.")

    @abstractmethod
    def _forward_batch(
        self, net: BaseNet, data: dict[str, Any], mode: Literal["train", "val", "test"]
    ) -> dict[str, Any]:
        """Compute loss, preds and other statistics for different modes."""
        raise NotImplementedError("_forward_batch method must be implemented.")

    @abstractmethod
    def _post_process(
        self,
        batch_idx: int,
        res: dict[str, Any],
        batch: dict[str, Any],
        data: dict[str, Any],
        metrics: dict[str, Any],
        statistics: list,
        info: dict[str, Any],
        mode: Literal["train", "val", "test"],
    ) -> None:
        """
        Post-process batch results, including compute metrics, collect statistics and record information.

        Note:
            metrics: Used to record and update the loss and other metrics in real time for each batch, and to output
            them to the console.
            statistics: Used to record the statistical data for each batch, and for calculating the global indicators
            at the end of each round.
            info: Used to record other data in round, and print the summary by trainer at the end of the round.
        """
        raise NotImplementedError("_post_batch method must be implemented.")

    @abstractmethod
    def write(self, batches: int, writer: SummaryWriter, res: dict[str, Any], metrics: dict[str, Any]) -> None:
        """Focus on the recording of batch-based data, while epoch-based data is recorded by the trainer."""
        raise NotImplementedError("write method must be implemented.")

    @torch.no_grad()
    def validate(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Returns val metrics and info dict for the epoch."""
        self.set_eval()

        info = {}
        statistics = []
        metrics = self._init_metrics("val")
        self.print_metric_titles("val", metrics)

        pbar = tqdm(enumerate(self.val_loader), total=self.val_batches)
        for i, batch in pbar:
            data = self._prepare_batch(batch, "val")
            net = self.net if not self.ema_enable else self.emas["net"].ema
            res = self._forward_batch(net, data, "val")

            self._post_process(i, res, batch, data, metrics, statistics, info, "val")
            self.pbar_log("val", pbar, **metrics)

        return metrics, info

    @torch.no_grad()
    def eval(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Evaluate the preformece of the model on test dataset.

        Note:
            Verify the objective metrics (mAP, mIoU, Loss) of the model on the test set.
        """
        self.set_eval()
        if self.test_loader is None:
            raise ValueError("Test dataloader is not available.")

        info = {}
        statistics = []
        metrics = self._init_metrics("test")
        self.print_metric_titles("test", metrics)

        pbar = tqdm(enumerate(self.test_loader), total=self.test_batches)
        for i, batch in pbar:
            data = self._prepare_batch(batch, "test")
            res = self._forward_batch(self.net, data, "test")

            self._post_process(i, res, batch, data, metrics, statistics, info, "test")
            self.pbar_log("test", pbar, **metrics)

        return metrics, info

    def backward(self, loss: torch.Tensor) -> None:
        if self.accumulate > 1:
            loss = loss / self.accumulate

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(self, batches: int) -> None:
        if batches - self.last_opt_step >= self.accumulate:
            if self.scaler is not None:
                # unscale gradients, necessary if gradient clipping is performed after.
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg["optimizer"]["grad_clip"])

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.ema_enable:
                for key, ema in self.emas.items():
                    ema.update(self.nets[key])  # Update EMA

            self.optimizer.zero_grad()
            self.last_opt_step = batches

    @abstractmethod
    def predict(self, streams: str | StreamBase, *args, **kwargs) -> None:
        """
        Make predictions from different special data stream.

        Note:
            Deploy the model to the real world, observe the actual operation effect and output the results.
        """
        self.set_eval()

    def save(
        self, epoch: int, val_info: dict, best_fitness: float, save_path: str, record_dir: str, ckpt_dir: str
    ) -> None:
        """Save checkpoint."""
        state = {
            "epoch": epoch,
            "cfg": self.cfg,
            "best_fitness": best_fitness,
            "nets": {},
            "optimizers": {},
            "last_opt_step": self.last_opt_step,
            "amp": self.amp_enable,
            "record_dir": record_dir,
            "ckpt_dir": ckpt_dir,
            "emas": None,
        }

        for key, val in val_info.items():
            state[key] = val

        # save the nets' parameters
        for key, net in self.nets.items():
            state["nets"][key] = net.state_dict()

        # save the optimizers' parameters
        for key, optimizer in self.optimizers.items():
            state["optimizers"][key] = optimizer.state_dict()

        # save the schedulers' parameters
        if hasattr(self, "schedulers"):
            state["schedulers"] = {k: v.state_dict() for k, v in self.schedulers.items()}

        # save the scaler states
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()

        # save ema
        if self.ema_enable:
            emas = {}
            for key, ema in self.emas.items():
                emas[key] = {
                    "model_state": ema.ema.state_dict(),
                    "updates": ema.updates,
                }
            state["emas"] = emas

        torch.save(state, save_path)
        LOGGER.info(f"Saved checkpoint to {save_path}.")

    def load(self, checkpoint: str, load_ema: bool = False) -> dict[str, Any]:
        """
        Load checkpoint.

        Args:
            checkpoint (str): The path of checkpoint.
            load_ema (bool): Whether to load ema parameters to nets for eval mode. Defaults to False.
        Returns:
            dict: The state dict loaded from checkpoint.
        """
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)

        # cfg
        self._cfg = state["cfg"]  # include dataset, trainer

        # trainer cfg and dataset cfg
        dataset_cfg = self._cfg["data"].get("dataset", {})
        if hasattr(self, "_dataset_cfg"):
            dataset_cfg.update(self.dataset_cfg)
        self._dataset_cfg = dataset_cfg

        trainer_cfg = self._cfg.get("trainer", {})
        if hasattr(self, "_trainer_cfg"):
            trainer_cfg.update(self.trainer_cfg)
        self._trainer_cfg = trainer_cfg

        self.last_opt_step = state.get("last_opt_step", -1)

        # load the nets' parameters
        if load_ema and state.get("emas") is not None:
            for key, net in self.nets.items():
                if key in state["emas"]:
                    ema_state = state["emas"][key]
                    if isinstance(ema_state, dict) and "model_state" in ema_state:
                        net.load_state_dict(ema_state["model_state"], strict=True)
                        LOGGER.info(f"Loaded EMA parameters into network '{key}' for evaluation.")
                    else:
                        LOGGER.warning(f"Invalid EMA state format for network '{key}', loading normal weights.")
                        if key in state["nets"]:
                            net.load_state_dict(state["nets"][key], strict=True)
                else:
                    LOGGER.warning(f"EMA for network '{key}' not found in checkpoint, loading normal weights.")
                    if key in state["nets"]:
                        net.load_state_dict(state["nets"][key], strict=True)

        else:
            for key, net in self.nets.items():
                net.load_state_dict(state["nets"][key], strict=True)

        # load the optimizers' parameters
        for key, optimizer in self.optimizers.items():
            if key in state.get("optimizers", {}):
                try:
                    optimizer.load_state_dict(state["optimizers"][key])
                    LOGGER.info(f"Loaded optimizer '{key}' state.")
                except Exception as e:
                    LOGGER.warning(f"Error loading optimizer '{key}': {e}.")
            else:
                LOGGER.warning(f"Optimizer '{key}' not found in checkpoint.")

        # load the schedulers' parameters
        if hasattr(self, "schedulers") and "schedulers" in state:
            for key, scheduler in self.schedulers.items():
                if key in state["schedulers"]:
                    try:
                        scheduler.load_state_dict(state["schedulers"][key])
                        LOGGER.info(f"Loaded scheduler '{key}' state.")
                    except Exception as e:
                        LOGGER.warning(f"Error loading scheduler '{key}': {e}.")
                else:
                    LOGGER.info(f"Scheduler '{key}' not found in checkpoint.")

        if "scaler" in state:
            if self.amp_enable:
                self.scaler.load_state_dict(state["scaler"])
                LOGGER.info("Loaded AMP scaler state.")
            else:
                LOGGER.warning("Checkpoint has AMP scaler but current AMP is disabled, ignoring scaler.")
        elif self.amp_enable:
            LOGGER.warning("AMP enabled but checkpoint has no scaler, use new scaler.")
            if self.scaler is None:
                self.scaler = GradScaler()

        # load ema
        if self.ema_enable:
            if state.get("emas") is None:
                LOGGER.warning("EMA enabled but checkpoint has no EMA states, Creating new EMAs.")
                self._init_ema()
            else:
                for key, ema in self.emas.items():
                    if key in state["emas"]:
                        ema_state = state["emas"][key]
                        # restore ema state
                        ema.ema.load_state_dict(ema_state["model_state"])
                        ema.updates = ema_state.get("updates", 0)
                        LOGGER.info(f"Loaded EMA for network '{key}' (updates: {ema.updates}).")

        LOGGER.info(f"Successfully loaded checkpoint from {checkpoint}.")
        if "epoch" in state:
            LOGGER.info(f"Checkpoint epoch: {state['epoch']}")
        if "best_fitness" in state:
            LOGGER.info(f"Checkpoint best_fitness: {state['best_fitness']:.4f}")

        return state
