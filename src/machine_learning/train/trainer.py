from typing import Any, Union, Mapping
import os
import torch
from tqdm import trange

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from .trainer_cfg import TrainCfg
from machine_learning.algorithms import AlgorithmBase
from machine_learning.utils.others import set_seed


class Trainer:
    def __init__(
        self,
        cfg: TrainCfg,
        parsed_data: Mapping[str, Union[Dataset, Any]],
        algo: AlgorithmBase,
    ) -> None:
        """
        The trainer of all machine learning algorithm

        Args:
            cfg (TrainCfg): The configuration of the trainer.
            parsed_data (Mapping[str, Union[Dataset, Any]]): Parsed specific dataset data, must including train_dataset and
            val_dataset, may contain data information of the specific dataset.
            algo (AlgorithmBase): The algorithm to be trained.
        """
        self.cfg = cfg
        self._algorithm = algo
        print(self.cfg.epochs)
        # ------------------ configure global random seed -----------------
        set_seed(self.cfg.seed)
        print(f"[INFO] Current seed: {self.cfg.seed}")

        # ---------------------- configure algo data ----------------------
        self.train_batch_size = self.cfg.train_batch_size
        self.val_batch_size = self.cfg.val_batch_size
        self.test_batch_size = self.cfg.test_batch_size
        self._configure_data(parsed_data)

        # --------------------- configure algo logger ---------------------
        self._configure_writer()
        self.best_loss = torch.inf

    def _configure_data(self, data: Mapping[str, Union[Dataset, Any]]) -> None:
        _necessary_key_type_couples_ = {"train_dataset": Dataset, "val_dataset": Dataset}

        for key, type in _necessary_key_type_couples_.items():
            if key not in data or not isinstance(data[key], type):
                raise ValueError(f"Input data mapping has no {key} or {key} is not Dataset type.")

        train_dataset, val_dataset = data.pop("train_dataset"), data.pop("val_dataset")

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.cfg.data_shuffle,
            num_workers=self.cfg.data_num_workers,
            collate_fn=train_dataset.collate_fn if hasattr(train_dataset, "collate_fn") else None,
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.data_num_workers,
            collate_fn=val_dataset.collate_fn if hasattr(val_dataset, "collate_fn") else None,
        )

        test_loader = None
        if "test_dataset" in data and isinstance(data["test_dataset"], Dataset):
            test_dataset = data.pop("test_dataset")
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=self.cfg.data_num_workers,
                collate_fn=test_dataset.collate_fn if hasattr(test_dataset, "collate_fn") else None,
            )

        self._algorithm._initialize_dependent_on_data(
            batch_size=self.cfg.train_batch_size,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            **data,
        )

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
            train_loss = self._algorithm.train_epoch(epoch, self.writer, self.cfg.log_interval)
            val_loss = self._algorithm.validate()

            # adjust the learning rate
            if self._algorithm._schedulers:
                for key, val in self._algorithm._schedulers.items():
                    if isinstance(val, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        val.step(val_loss[key])
                    else:
                        val.step()

            # log the train loss
            for key, val in train_loss.items():
                self.writer.add_scalar(f"{key} loss/train", val, epoch)

            # log the val loss
            for key, val in val_loss.items():
                self.writer.add_scalar(f"{key} loss/val", val, epoch)

            # save the best model
            # must set the best_model option to True in train_cfg and return "save" loss item in val loss dict in algo
            if self.cfg.save_best and "save" in val_loss:
                if val_loss["save"] < self.best_loss:
                    self.best_loss = val_loss["save"]
                    self.save_checkpoint(epoch, val_loss, self.best_loss, is_best=True)
            else:
                print("Saving of the best loss model skipped.")

            # save the model regularly
            if self.cfg.save_interval and (epoch + 1) % self.cfg.save_interval == 0:
                self.save_checkpoint(epoch, val_loss, self.best_loss, is_best=False)

            # print log information
            print(f"Epoch: {epoch + 1:03d} | ", end="")
            for key, val in train_loss.items():
                print(f"{key} train loss {val:.4f} | ", end="")
            for key, val in val_loss.items():
                if key != "save":
                    print(f"{key} val loss {val:.4f} | ", end="")
                else:
                    print(f"{key} loss {val:.4f} | ", end="")
            for key, opt in self._algorithm._optimizers.items():
                print(f"{key} lr: {opt.param_groups[0]['lr']:.2e} | ")

    def train_from_checkpoint(self, checkpoint: str) -> None:
        state_dict = self.load(checkpoint)
        start_epoch = state_dict["epoch"]
        self.train(start_epoch)

    def eval(self, *args, **kwargs):
        self._algorithm.eval(*args, **kwargs)

    def save_checkpoint(self, epoch: int, loss: dict, best_loss: float, is_best: bool = False) -> None:
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

        self._algorithm.save(epoch, loss, best_loss, save_path)

    def load(self, checkpoint: str) -> dict[str:Any]:
        return self._algorithm.load(checkpoint)
