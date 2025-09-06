from typing import Any
import os
import torch
from datetime import datetime
from prettytable import PrettyTable
from numbers import Integral, Real

from torch.utils.tensorboard import SummaryWriter

from machine_learning.utils.cfg import BaseCfg
from machine_learning.utils.logger import LOGGER
from dataclasses import dataclass, field, MISSING
from machine_learning.algorithms import AlgorithmBase
from machine_learning.utils import set_seed, print_cfg, cfg_to_dict


@dataclass
class TrainCfg(BaseCfg):
    log_dir: str = MISSING
    model_dir: str = MISSING
    seed: int = field(default=23)
    epochs: int = field(default=100)
    log_interval: int = field(default=10)
    save_interval: int = field(default=10)
    save_best: bool = field(default=True)


class Trainer:
    def __init__(
        self,
        cfg: TrainCfg,
        algo: AlgorithmBase,
    ) -> None:
        """
        The trainer of all machine learning algorithm

        Args:
            cfg (TrainCfg): The configuration of the trainer.
            algo (AlgorithmBase): The algorithm to be trained.
        """
        self.cfg = cfg
        self._algorithm = algo

        self.epochs = self.cfg.epochs

        # datetime suffix for logging
        self.dt_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # ------------------ configure global random seed ------------------
        set_seed(self.cfg.seed)
        LOGGER.info(f"Current seed: {self.cfg.seed}")

        # ------------------------ initilaize algo -------------------------
        self._algorithm._add_cfg("train", cfg_to_dict(self.cfg))
        self._algorithm._initialize()
        print_cfg("Configuration", self._algorithm.cfg)

        # ------------------------ configure writer ------------------------
        self._configure_writer()
        self.best_loss = torch.inf

    @property
    def algorithm(self) -> AlgorithmBase:
        return self._algorithm

    def _configure_writer(self):
        log_path = self.cfg.log_dir + self.dt_suffix
        log_path = os.path.abspath(log_path)

        try:
            os.makedirs(log_path, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create log directory at {log_path}: {e}")

        self.writer = SummaryWriter(log_dir=log_path)

    def _setup_train(self):
        # reset optimizer gradients to zeros
        for opt in self.algorithm.optimizers.values():
            opt.zero_grad()

    def train(self, start_epoch: int = 0) -> None:
        """Train the algorithm"""
        LOGGER.info(f"Start training for {self.epochs} epochs...")

        self._setup_train()

        for epoch in range(start_epoch, self.cfg.epochs):
            train_metrics = self.algorithm.train_epoch(epoch, self.writer, self.cfg.log_interval)
            val_metrics = self.algorithm.validate()

            # adjust the learning rate
            if len(self.algorithm.schedulers) == 1:  # single net, single optimizer
                scheduler = self.algorithm.schedulers["scheduler"]
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["vloss"])
                else:
                    scheduler.step()
            elif len(self.algorithm.schedulers) > 1:  # multi nets, multi optimizers
                for name, scheduler in self.algorithm.schedulers.items():
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_metrics[name + "loss"])
                    else:
                        scheduler.step()

            # log the train loss
            for key, val in train_metrics.items():
                if not isinstance(val, (Real, Integral)):
                    continue
                self.writer.add_scalar(f"{key}/train", val, epoch)

            # log the val loss
            for key, val in val_metrics.items():
                if (key == "sloss") or (not isinstance(val, (Real, Integral))):
                    continue
                self.writer.add_scalar(f"{key}/val", val, epoch)

            # save the best model
            # must set the best_model option to True in train_cfg and return "save" loss item in val loss dict in algo
            if self.cfg.save_best and "sloss" in val_metrics:
                if val_metrics["sloss"] < self.best_loss:
                    self.best_loss = val_metrics["sloss"]
                    self.save_checkpoint(epoch, val_metrics, self.best_loss, is_best=True)
            else:
                LOGGER.info("Saving of the best loss model skipped.")

            # save the model regularly
            if self.cfg.save_interval and (epoch + 1) % self.cfg.save_interval == 0:
                self.save_checkpoint(epoch, val_metrics, self.best_loss, is_best=False)

            # print epoch information
            if epoch != self.cfg.epochs:
                self.epoch_info()
            else:
                self.epoch_info(val_metrics)

    def train_from_checkpoint(self, checkpoint: str) -> None:
        state_dict = self._algorithm.load(checkpoint)
        start_epoch = state_dict["epoch"] + 1
        self.best_loss = state_dict.get("best_loss", float("inf"))

        self.train(start_epoch)

    def save_checkpoint(self, epoch: int, val_return: dict, best_loss: float, is_best: bool = False) -> None:
        model_path = self.cfg.model_dir + self.dt_suffix
        model_path = os.path.abspath(model_path)

        try:
            os.makedirs(model_path, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create log directory at {model_path}: {e}")

        filename = f"checkpoint_epoch_{epoch + 1}.pth"
        if is_best:
            filename = "best_model.pth"
        save_path = os.path.join(model_path, filename)

        self._algorithm.save(epoch, val_return, best_loss, save_path)

    def epoch_info(
        self,
        epoch: int,
        trian_metrics: dict[Any] | None = None,
        val_metrics: dict[Any] | None = None,
    ) -> None:
        log_table = PrettyTable()
        log_table.title = f"Epoch info:{epoch + 1}"
        log_table.field_names = ["metrics", "Value"]

        rows = []
        if trian_metrics:
            for key, val in trian_metrics.items():
                rows.append([f"train: {key}", val])
        if val_metrics:
            for key, val in val_metrics.items():
                rows.append([f"train: {key}", val])
        for key, opt in self._algorithm._optimizers.items():
            rows.append([f"{key} lr", opt.param_groups[0]["lr"]])

        log_table.add_rows(rows)
        LOGGER.info("\n" + log_table.get_string())
