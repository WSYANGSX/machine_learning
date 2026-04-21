from typing import Any, Mapping

import os
import json
import yaml
import torch
from datetime import datetime
from numbers import Integral, Real
from torch.utils.tensorboard import SummaryWriter

from rich import box
from rich.json import JSON
from rich.table import Table
from rich.console import Console

from .trianer_cfg import TrainerCfg
from machine_learning.utils.logger import LOGGER
from machine_learning.algorithms import AlgorithmBase, global_factory
from machine_learning.utils.constants import ROOT_PATH, ALGOCFG_PATH, DATACFG_PATH
from machine_learning.utils import set_seed, cfg2dict, print_cfg, load_cfg


class Trainer:
    def __init__(
        self,
        name: str,
        train_cfg: TrainerCfg,
        algorithm_cfg: str | Mapping[str, Any],
        dataset_cfg: str | Mapping[str, Any],
    ) -> None:
        """
        The trainer of all machine learning algorithm

        Args:
            name (str): The name of the algorithm to train.
            train_cfg (TrainCfg): The configuration of the trainer.
            algorithm (str, Mapping[str, Any]): The algorithm cfg.
            dataset (str, Mapping[str, Any]): The dataset cfg.
            amp (bool): Whether to enable Automatic Mixed Precision during training. Defaults to False.
            ema (bool): Whether to enable Exponential Moving Average during training. Defaults to False.
            device (Literal["cuda", "cpu", "auto"]): Running device. Defaults to "auto"-automatic selection.
            continue_training (bool): Whether to continue training from a checkpoint. Defaults to False.
            ckpt(str): The path of the checkpoint to continue training. Defaults to None.
        """
        self.continue_training = train_cfg.continue_training

        if self.continue_training:
            if train_cfg.ckpt is None:
                raise ValueError("Checkpoint must be provided when continuing training.")

            LOGGER.info(f"Continuing training from checkpoint: {train_cfg.ckpt}")

            algo_cfg = load_cfg(os.path.join(train_cfg.ckpt, "config.yaml"))
            self.cfg = algo_cfg["trainer"]

            self.record_dir = self.cfg["record_dir"]
            self.ckpt_dir = self.cfg["ckpt_dir"]

        else:
            self.cfg = cfg2dict(train_cfg)

            # load cfg
            algo_cfg = self._load_alogrithm_cfg(algorithm_cfg)
            dataset_cfg = self._load_datasetcfg(dataset_cfg)

            # datetime suffix for log
            self.dt_suffix = (
                name
                + "_"
                + (algo_cfg["net"]["net_scale"] if "net_scale" in algo_cfg["net"] else "")
                + "_"
                + (os.path.splitext(dataset_cfg)[0] if isinstance(dataset_cfg, str) else dataset_cfg["name"])
                + "_"
                + datetime.now().strftime("%Y-%m-%d_%H-%M")
            )

            self.record_dir = os.path.abspath(ROOT_PATH + "/runs/" + f"{name}/" + self.dt_suffix)
            self.ckpt_dir = os.path.abspath(self.record_dir + "/ckpt")
            self.cfg["record_dir"] = self.record_dir  # add record path to cfg
            self.cfg["ckpt_dir"] = self.ckpt_dir  # add ckpt path to cfg

            # add train_cfg and dataset_cfg to algo
            algo_cfg["trainer"] = self.cfg
            algo_cfg["data"]["dataset"] = dataset_cfg

        self.epochs = self.cfg["epochs"]
        amp = self.cfg["amp"]
        ema = self.cfg["ema"]
        device = self._configure_device(self.cfg["device"])
        seed = self.cfg["seed"]

        # ------------------ init global random seed ------------------
        set_seed(seed)
        LOGGER.info(f"Global seed: {seed}")

        # --------------------- build algorithm ---------------------
        self._build_algorithm(name, device, algo_cfg, amp, ema)
        print_cfg("Total configuration", algo_cfg)

        # --------------------- init writer ---------------------------
        self._init_writer(self.record_dir)
        if not self.continue_training:
            # record algorithm cfg
            with open(self.record_dir + "/config.yaml", "w", encoding="utf-8") as file:
                yaml.dump(algo_cfg, file, default_flow_style=False, allow_unicode=True)

        # set best_fitness value for saving the best ckpt
        self.task = algo_cfg["algorithm"].get("task", "")
        if self.task in ("detect", "segment"):  # mAP, mIOU
            self.best_fitness = 0.0
        else:
            self.best_fitness = torch.inf

    @property
    def algorithm(self) -> AlgorithmBase:
        return self._algorithm

    def _load_alogrithm_cfg(self, cfg: str | Mapping[str, Any]) -> dict:
        """Load algorithm cfg from file or dict."""
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

    def _configure_device(self, device: str) -> torch.device:
        """Configure the trianing device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _build_algorithm(
        self,
        name: str,
        device: torch.device,
        cfg: Mapping[str, Any],
        amp: bool | None = None,
        ema: bool | None = False,
    ) -> None:
        LOGGER.info(f"Building algorithm {name}...")
        self._algorithm = global_factory.create_algorithm(
            algo=name, cfg=cfg, name=name, device=device, amp=amp, ema=ema
        )
        self._algorithm._init_on_trainer()

    def _init_writer(self, path: str):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create record directory at {path}: {e}")

        self.writer = SummaryWriter(log_dir=path)

    def _setup_train(self):
        # reset optimizer gradients to zeros
        for opt in self.algorithm.optimizers.values():
            opt.zero_grad()

    def _train(self, start_epoch: int = 0) -> None:
        """Train the algorithm"""
        LOGGER.info(f"Start training for {self.epochs - start_epoch} epochs...")
        strat_time = datetime.now()

        self._setup_train()

        for epoch in range(start_epoch, self.epochs):
            train_metrics, train_info = self.algorithm.train_epoch(epoch, self.writer, self.cfg["log_interval"])
            val_metrics, val_info = self.algorithm.validate()

            # adjust the learning rate
            if len(self.algorithm.schedulers) == 1:  # single net, single optimizer
                scheduler = self.algorithm.schedulers["scheduler"]
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Give priority to using save best. If not available, degrade to vloss
                    monitor_metric = val_metrics.get("best_fitness", val_metrics.get("vloss"))
                    scheduler.step(monitor_metric)
                else:
                    scheduler.step()
            elif len(self.algorithm.schedulers) > 1:  # multi nets, multi optimizers
                for name, scheduler in self.algorithm.schedulers.items():
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        # For multiple networks, usually their respective losses or the global save best are used
                        monitor_metric = val_metrics.get("best_fitness", val_metrics.get(name + "loss"))
                        scheduler.step(monitor_metric)
                    else:
                        scheduler.step()

            # log the train loss
            for key, val in train_metrics.items():
                if not isinstance(val, (Real, Integral)):
                    continue
                self.writer.add_scalar(f"{key}/train", val, epoch)

            # log the val loss
            for key, val in val_metrics.items():
                if not isinstance(val, (Real, Integral)):
                    continue
                self.writer.add_scalar(f"{key}/val", val, epoch)

            # log lr
            if hasattr(self._algorithm, "optimizers"):
                for name, opt in self._algorithm.optimizers.items():
                    for i, param_group in enumerate(opt.param_groups):
                        lr_val = param_group.get("lr", "N/A")
                        self.writer.add_scalar(f"{name}: lr_pg{i}/train", lr_val, epoch)

            # save the best model
            # must set the best_model option to True in train_cfg and return "save_indicator" item in val method in algo
            if self.cfg["save_best"] and "best_fitness" in val_metrics:
                current_best_fitness = val_metrics["best_fitness"]
                if self.task in ("detect", "segment"):
                    if current_best_fitness > self.best_fitness:  # mAP, mIOU, ... as references
                        self.best_fitness = current_best_fitness
                        self.save_checkpoint(epoch, val_metrics, self.best_fitness, is_best=True)
                else:  # Vloss as a reference
                    if current_best_fitness < self.best_fitness:
                        self.best_fitness = current_best_fitness
                        self.save_checkpoint(epoch, val_metrics, self.best_fitness, is_best=True)
            else:
                LOGGER.info("Saving of the model with the best fitness skipped.")

            # save the model regularly
            if self.cfg["save_interval"] and (epoch + 1) % self.cfg["save_interval"] == 0:
                self.save_checkpoint(epoch, val_metrics, self.best_fitness, is_best=False)

            # print epoch information
            if epoch == self.epochs - 1:
                self.epoch_info(epoch=epoch, val_info=val_info)

                # Print training completion time
                total_time = datetime.now() - strat_time
                total_hours = total_time.total_seconds() / 3600
                LOGGER.info(f"Training completed in: {total_hours:.2f} hours.")

            else:
                self.epoch_info(epoch=epoch)

    def train(self) -> None:
        if self.continue_training:
            last_model = os.path.join(self.ckpt_dir, "last_model.pth")
            state_dict = self._algorithm.load(last_model)
            start_epoch = state_dict["epoch"] + 1
            self.best_fitness = state_dict.get("best_fitness", self.best_fitness)
            self._train(start_epoch)
        else:
            self._train()

    def save_checkpoint(self, epoch: int, val_return: dict, best_fitness: float, is_best: bool = False) -> None:
        try:
            os.makedirs(self.ckpt_dir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create ckpt directory at {self.ckpt_dir}: {e}")

        filename = f"checkpoint_epoch_{epoch + 1}.pth"
        if is_best:
            filename = "best_model.pth"
        save_path = os.path.join(self.ckpt_dir, filename)

        self._algorithm.save(epoch, val_return, best_fitness, save_path, self.record_dir, self.ckpt_dir)

        # save last model
        last_path = os.path.join(self.ckpt_dir, "last_model.pth")
        self._algorithm.save(epoch, val_return, best_fitness, last_path, self.record_dir, self.ckpt_dir)

    def epoch_info(
        self,
        epoch: int,
        train_metrics: dict[str, Any] | None = None,
        val_metrics: dict[str, Any] | None = None,
        train_info: dict[Any] | None = None,
        val_info: dict[Any] | None = None,
    ) -> None:
        """
        Print the information of the current epoch using Rich Table.
        """
        console = Console()

        table = Table(
            title=f"Epoch info: {epoch + 1} Summary",
            header_style="bold magenta",
            show_header=True,
            box=box.ROUNDED,
            show_lines=True,
            title_style="bold blue not italic",
        )

        table.add_column("Category", style="cyan", justify="center", vertical="middle")
        table.add_column("Item", style="dim", justify="center", vertical="middle")
        table.add_column("Value", justify="center", vertical="middle")

        def _format_value(val):
            if isinstance(val, (float, int)):
                # if the float is very small, use scientific notation
                if isinstance(val, float) and (val < 1e-4 and val > 0):
                    return f"{val:.4e}"
                if isinstance(val, float):
                    return f"{val:.8f}"
                return str(val)
            elif isinstance(val, (dict, list)):
                return JSON(json.dumps(val))
            else:
                return str(val)

        if hasattr(self._algorithm, "optimizers"):
            for name, opt in self._algorithm.optimizers.items():
                for i, param_group in enumerate(opt.param_groups):
                    lr_val = param_group.get("lr", "N/A")
                    table.add_row("lr", f"{name}/(pg{i})", _format_value(lr_val))

        if train_metrics:
            for k, v in train_metrics.items():
                table.add_row("Train Metric", k, _format_value(v))

        if val_metrics:
            for k, v in val_metrics.items():
                style = "bold green" if k == "best_fitness" else None
                val_str = _format_value(v)
                if style:
                    pass
                table.add_row("Val Metric", k, val_str)

        if train_info:
            for k, v in train_info.items():
                table.add_row("Train Info", k, _format_value(v))

        if val_info:
            for k, v in val_info.items():
                table.add_row("Val Info", k, _format_value(v))

        print("\n")
        console.print(table)
