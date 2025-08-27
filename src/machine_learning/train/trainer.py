from typing import Any, Mapping, Union
import os
import torch
from tqdm import trange
from prettytable import PrettyTable

from torch.utils.data import Dataset
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
    amp: bool = field(default=False)


class Trainer:
    def __init__(
        self,
        cfg: TrainCfg,
        algo: AlgorithmBase,
        data: Mapping[str, Union[Dataset, Any]],
    ) -> None:
        """
        The trainer of all machine learning algorithm

        Args:
            cfg (TrainCfg): The configuration of the trainer.
            algo (AlgorithmBase): The algorithm to be trained.
            data (Mapping[str, Union[Dataset, Any]]): Parsed specific dataset data, must including train dataset and val
            dataset, may contain data information of the specific dataset.
        """
        self.cfg = cfg
        self._algorithm = algo

        self.amp = self.cfg.amp
        self.epochs = self.cfg.epochs

        # ------------------ configure global random seed ------------------
        set_seed(self.cfg.seed)
        LOGGER.info(f"Current seed: {self.cfg.seed}")

        # ------------------------ initilaize algo -------------------------
        self._algorithm._add_cfg("train", cfg_to_dict(self.cfg))
        self._algorithm._initialize(data=data, amp=self.amp)
        print_cfg("Configuration", self._algorithm.cfg)

        # ------------------------ configure writer ------------------------
        self._configure_writer()
        self.best_loss = torch.inf

    @property
    def algorithm(self) -> AlgorithmBase:
        return self._algorithm

    def _configure_writer(self):
        log_path = self.cfg.log_dir
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
        LOGGER.info("Start training...")

        self._setup_train()

        for epoch in trange(start_epoch, self.cfg.epochs):
            train_res = self.algorithm.train_epoch(epoch, self.writer, self.cfg.log_interval)
            val_res = self.algorithm.validate()

            # adjust the learning rate
            if len(self.algorithm.schedulers) == 1:  # single net, single optimizer
                scheduler = self.algorithm.schedulers["scheduler"]
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_res["loss"])
                else:
                    scheduler.step()
            elif len(self.algorithm.schedulers) > 1:  # multi nets, multi optimizers
                for name, scheduler in self.algorithm.schedulers.items():
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_res[name + " loss"])
                    else:
                        scheduler.step()

            # log the train loss
            for key, val in train_res.items():
                self.writer.add_scalar(f"{key}/train", val, epoch)

            # log the val loss
            for key, val in val_res.items():
                if key == "save metric":
                    continue
                self.writer.add_scalar(f"{key}/val", val, epoch)

            # save the best model
            # must set the best_model option to True in train_cfg and return "save" loss item in val loss dict in algo
            if self.cfg.save_best and "save metric" in val_res:
                if val_res["save metric"] < self.best_loss:
                    self.best_loss = val_res["save metric"]
                    self.save_checkpoint(epoch, val_res, self.best_loss, is_best=True)
            else:
                LOGGER.info("Saving of the best loss model skipped.")

            # save the model regularly
            if self.cfg.save_interval and (epoch + 1) % self.cfg.save_interval == 0:
                self.save_checkpoint(epoch, val_res, self.best_loss, is_best=False)

            # print log information
            self.log_epoch_info(epoch, train_res, val_res)

    def train_from_checkpoint(self, checkpoint: str) -> None:
        state_dict = self._algorithm.load(checkpoint)
        start_epoch = state_dict["epoch"] + 1
        self.best_loss = state_dict.get("best_loss", float("inf"))

        self.train(start_epoch)

    def save_checkpoint(self, epoch: int, val_return: dict, best_loss: float, is_best: bool = False) -> None:
        model_path = self.cfg.model_dir
        model_path = os.path.abspath(model_path)

        try:
            os.makedirs(model_path, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create log directory at {model_path}: {e}")

        filename = f"checkpoint_epoch_{epoch}.pth"
        if is_best:
            filename = "best_model.pth"
        save_path = os.path.join(model_path, filename)

        self._algorithm.save(epoch, val_return, best_loss, save_path)

    def log_epoch_info(self, epoch: int, train_info: dict[Any], val_info: dict[Any]) -> None:
        log_table = PrettyTable()
        log_table.title = "Evaluation indicators: " + f"Epoch {epoch}"
        log_table.field_names = ["Indicator", "Value"]

        rows = []
        for key, val in train_info.items():
            rows.append([f"Train: {key}", val])
        for key, val in val_info.items():
            rows.append([f"Val: {key}", val])
        for key, opt in self._algorithm._optimizers.items():
            rows.append([f"{key.capitalize()} learning rate", opt.param_groups[0]["lr"]])

        log_table.add_rows(rows)
        print(log_table)
