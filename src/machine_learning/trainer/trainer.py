from typing import Any, Mapping

import os
import yaml
import torch
from datetime import datetime
from prettytable import PrettyTable
from numbers import Integral, Real
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from .trian_cfg import TrainCfg
from machine_learning.utils.logger import LOGGER
from machine_learning.algorithms import AlgorithmBase
from machine_learning.utils.constants import DATACFG_PATH
from machine_learning.dataset import ParserBase, PARSER_MAPS, DATASET_MAPS
from machine_learning.utils import set_seed, cfg2dict, print_cfg


class Trainer:
    def __init__(
        self,
        cfg: TrainCfg,
        dataset: str | Mapping[str, Any],
        algo: AlgorithmBase,
    ) -> None:
        """
        The trainer of all machine learning algorithm

        Args:
            cfg (TrainCfg): The configuration of the trainer.
            dataset (str, Mapping[str, Any]): The dataset cfg.
            algo (AlgorithmBase): The algorithm to be trained.
        """
        self.cfg = cfg
        self._algorithm = algo

        self.epochs = self.cfg.epochs

        # datetime suffix for ckpt
        self.dt_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.ckpt_dir = os.path.abspath(self.cfg.ckpt_dir + self.dt_suffix)

        # ------------------ init global random seed ------------------
        set_seed(self.cfg.seed)
        LOGGER.info(f"Current seed: {self.cfg.seed}")

        # --------------------- dataset cfg  --------------------------
        LOGGER.info("Parsing dataset cfg by trainer...")
        dataset_cfg = self.load_data_cfg(dataset)

        # -------------------- parse dataset --------------------------
        datasets = self.get_datasets(dataset_cfg)

        # ------------------- add cfg to algo -------------------------
        LOGGER.info("Algorithm initializing by trainer...")
        self.algorithm._init_on_trainer(cfg2dict(self.cfg), dataset_cfg, datasets)
        print_cfg("Total configuration", self.algorithm.cfg)

        # --------------------- init writer ---------------------------
        self._init_writer()
        self.best_loss = torch.inf

    @property
    def algorithm(self) -> AlgorithmBase:
        return self._algorithm

    def load_data_cfg(self, cfg: str | Mapping[str, Any]) -> dict:
        if isinstance(cfg, Mapping):
            cfg = dict(cfg)
        else:
            if not (os.path.splitext(cfg)[1] == ".yaml" or os.path.splitext(cfg)[1] == ".yml"):
                raise ValueError("Input path is not a yaml file path.")
            cfg_path = os.path.join(DATACFG_PATH, cfg)
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)

        return cfg

    def get_datasets(self, data_cfg: dict[str, Any]) -> dict[str, Dataset]:
        # get dataset parser
        dataset_name = data_cfg["name"]
        parser: ParserBase = PARSER_MAPS[dataset_name](data_cfg)

        # parser data
        data = parser.parse()
        trian_data, val_data, test_data = data["train"], data["val"], data.get("test", {})

        # build dataset
        data_type = data_cfg.get("type", None)

        if data_type is None:
            raise ValueError("The data type must be provided for Dataset mapping.")
        else:
            train_dataset = DATASET_MAPS[data_type](*trian_data, mode="train")
            val_dataset = DATASET_MAPS[data_type](*val_data, mode="val")
            if test_data:
                test_dataset = DATASET_MAPS[data_type](*test_data, mode="test")
            else:
                test_dataset = None

        return {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    def _init_writer(self):
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
        LOGGER.info(f"Start training for {self.epochs - start_epoch} epochs...")

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
            # must set the best_model option to True in train_cfg and return "sloss" item in val loss dict in algo
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
                self.epoch_info(epoch=epoch)
            else:
                self.epoch_info(epoch=epoch, val_metrics=val_metrics)

    def train_from_checkpoint(self, checkpoint: str) -> None:
        state_dict = self._algorithm.load(checkpoint)
        start_epoch = state_dict["epoch"] + 1
        self.best_loss = state_dict.get("best_loss", float("inf"))
        self.ckpt_dir = state_dict["ckpt_dir"]

        self.train(start_epoch)

    def save_checkpoint(self, epoch: int, val_return: dict, best_loss: float, is_best: bool = False) -> None:
        try:
            os.makedirs(self.ckpt_dir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create log directory at {self.ckpt_dir}: {e}")

        filename = f"checkpoint_epoch_{epoch + 1}.pth"
        if is_best:
            filename = "best_model.pth"
        save_path = os.path.join(self.ckpt_dir, filename)

        self._algorithm.save(epoch, val_return, best_loss, save_path, self.ckpt_dir)

    def epoch_info(
        self,
        epoch: int,
        trian_metrics: dict[Any] | None = None,
        val_metrics: dict[Any] | None = None,
    ) -> None:
        log_table = PrettyTable()
        log_table.title = f"Epoch info: {epoch + 1}"
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
