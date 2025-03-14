import os
import yaml
from typing import Literal
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision.transforms import Compose
from torch.utils.data import DataLoader

from machine_learning.utils import print_dict, CustomDataset


class AlgorithmBase(nn.Module, ABC):
    def __init__(
        self,
        config_file: str,
        name: str | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ):
        super().__init__()

        # -------------------- 设备配置 --------------------
        self.device = self._configure_device(device)

        # -------------------- 配置加载 --------------------
        self.config = self._load_config(config_file)
        self._validate_config()

        # -------------------- 设备配置 --------------------
        self.name = name if name is not None else self.config.get("algorithm", __class__.__name__)

        # -------------------- 数据记录 --------------------
        self._configure_writer()

    @property
    def algo_name(self) -> str:
        return self.name

    def _configure_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_config(self, config_file: str) -> dict:
        assert os.path.splitext(config_file)[1] == ".yaml" or os.path.splitext(config_file)[1] == ".yml", (
            "Please ultilize a yaml configuration file."
        )
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        print("Configuration parameters: ")
        print_dict(config)

        return config

    def _validate_config(self):
        """配置参数验证"""
        required_sections = ["data", "model", "training", "optimizer", "logging"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件中缺少必要部分: {section}")

    def _configure_writer(self):
        log_path = self.config["logging"].get(
            "log_dir",
            os.path.join(
                os.getcwd(),
                "logs",
                self.config.get("algorithm", __class__.__name__),
            ),
        )

        log_path = os.path.abspath(log_path)

        try:
            os.makedirs(log_path, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create log directory at {log_path}: {e}")

        self.writer = SummaryWriter(log_dir=log_path)
        self.best_loss = float("inf")

    def _load_datasets(
        self,
        train_data: torch.Tensor | np.ndarray,
        train_labels: torch.Tensor | np.ndarray,
        validate_data: torch.Tensor | np.ndarray,
        validate_labels: torch.Tensor | np.ndarray,
        transform: Compose,
    ):
        # 创建dataset和datasetloader
        train_dataset = CustomDataset(train_data, train_labels, transform)
        validate_dataset = CustomDataset(validate_data, validate_labels, transform)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
        )
        self.validate_loader = DataLoader(
            validate_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
        )

    @abstractmethod
    def _build_model(self):
        """
        构建算法中使用的模型
        """
        pass

    @abstractmethod
    def _configure_optimizer(self):
        """
        配置优化器
        """
        pass

    @abstractmethod
    def _configure_scheduler(self):
        """
        配置学习率调度器
        """
        pass
