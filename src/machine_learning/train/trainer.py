from typing import Any
import os
import torch
from tqdm import trange
from prettytable import PrettyTable

from torch.utils.tensorboard import SummaryWriter

from .trainer_cfg import TrainCfg
from machine_learning.algorithms import AlgorithmBase
from machine_learning.utils.others import set_seed


class Trainer:
    def __init__(self, cfg: TrainCfg, algo: AlgorithmBase) -> None:
        """
        The trainer of all machine learning algorithm

        Args:
            cfg (TrainCfg): The configuration of the trainer.
            algo (AlgorithmBase): The algorithm to be trained.
        """
        self.cfg = cfg
        self._algorithm = algo

        # ------------------ configure global random seed -----------------
        set_seed(self.cfg.seed)
        print(f"[INFO] Current seed: {self.cfg.seed}")

        # ------------------------ configure logger -----------------------
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

    def train(self, start_epoch: int = 0) -> None:
        """Train the algorithm"""
        print("[INFO] Start training...")

        for epoch in trange(start_epoch, self.cfg.epochs):
            train_return = self._algorithm.train_epoch(epoch, self.writer, self.cfg.log_interval)
            val_return = self._algorithm.validate()

            # adjust the learning rate
            if self._algorithm._schedulers:
                for key, val in self._algorithm._schedulers.items():
                    if isinstance(val, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        val.step(val_return[key + " loss"])
                    else:
                        val.step()

            # log the train loss
            for key, val in train_return.items():
                self.writer.add_scalar(f"{key}/train", val, epoch)

            # log the val loss
            for key, val in val_return.items():
                if key == "save metric":
                    continue
                self.writer.add_scalar(f"{key}/val", val, epoch)

            # save the best model
            # must set the best_model option to True in train_cfg and return "save" loss item in val loss dict in algo
            if self.cfg.save_best and "save metric" in val_return:
                if val_return["save metric"] < self.best_loss:
                    self.best_loss = val_return["save metric"]
                    self.save_checkpoint(epoch, val_return, self.best_loss, is_best=True)
            else:
                print("Saving of the best loss model skipped.")

            # save the model regularly
            if self.cfg.save_interval and (epoch + 1) % self.cfg.save_interval == 0:
                self.save_checkpoint(epoch, val_return, self.best_loss, is_best=False)

            # print log information
            self.log_epoch_info(epoch, train_return, val_return)

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
        log_table.title = "Evaluation indicators: " + f"Epoch {epoch + 1}"
        log_table.field_names = ["Indicator name", "Value"]

        rows = []
        for key, val in train_info.items():
            rows.append([f"Train: {key}", val])
        for key, val in val_info.items():
            rows.append([f"Val: {key}", val])
        for key, opt in self._algorithm._optimizers.items():
            rows.append([f"{key.capitalize()} learning rate", opt.param_groups[0]["lr"]])

        log_table.add_rows(rows)
        print(log_table)
