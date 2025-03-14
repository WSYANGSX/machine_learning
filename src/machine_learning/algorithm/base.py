import os
import yaml
from typing import Literal
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from machine_learning.utils import print_dict


class AlgorithmBase(nn.Module, ABC):
    def __init__(
        self,
        config_file: str,
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ):
        super().__init__()

        # -------------------- 设备配置 --------------------
        self.device = self._configure_device(device)

        # -------------------- 配置加载 --------------------
        self.config = self._load_config(config_file)
        self._validate_config()

        # -------------------- 数据记录 --------------------
        self._configure_writer()

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

    @abstractmethod
    def _build_model(self):
        NotImplementedError(f"Please implement the 'build_model' method for {self.__class__.__name__}.")

    @abstractmethod
    def _configure_optimizer(self):
        NotImplementedError(f"Please implement the 'build_model' method for {self.__class__.__name__}.")

    @abstractmethod
    def _configure_scheduler(self):
        NotImplementedError(f"Please implement the 'build_model' method for {self.__class__.__name__}.")

    @abstractmethod
    def _configure_transform(self):
        NotImplementedError(f"Please implement the 'build_model' method for {self.__class__.__name__}.")

    @abstractmethod
    def _load_datasets(self):
        NotImplementedError(f"Please implement the 'build_model' method for {self.__class__.__name__}.")
