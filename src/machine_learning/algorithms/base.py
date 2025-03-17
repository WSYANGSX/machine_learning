import os
import yaml
from typing import Literal, Mapping
from abc import ABC, abstractmethod

import torch
import torch.nn as nn  # noqa: F401


from machine_learning.models import BaseNet
from machine_learning.utils import print_dict


class AlgorithmBase(ABC):
    def __init__(
        self,
        cfg: str,
        models: Mapping[str, BaseNet],
        name: str | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ):
        """算法抽象基类

        Args:
            cfg (str): 算法配置, YAML文件路径.
            models (Mapping[str, BaseNet]): 算法所需的网络模型.
            name (str | None, optional): 算法名称. Defaults to None.
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): 算法运行设备. Defaults to "auto".
        """
        super().__init__()

        self._models = models

        # -------------------- 设备配置 --------------------
        self._device = self._configure_device(device)

        # -------------------- 配置加载 --------------------
        self._config = self._load_config(cfg)
        self._validate_config()

        # -------------------- 设备配置 --------------------
        self._name = name if name is not None else self.config.get("algorithm", __class__.__name__)

        # -------------------- 配置模型 --------------------
        self._configure_models()
        if self.config["model"]["initialize_weights"]:
            self._initialize_weights()

    @property
    def name(self) -> str:
        return self._name

    @property
    def models(self) -> list:
        return self._models.keys()

    @property
    def config(self) -> dict:
        return self._config

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
        required_sections = ["model", "optimizer", "scheduler"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件中缺少必要部分: {section}")

    def _configure_models(self):
        """
        配置模型
        """
        for model in self.models:
            model.to(self._device)

    def _initialize_weights(self) -> None:
        for model in self.models:
            model._initialize_weights()

    def _initialize_data_loader(self, train_data_loader, val_data_loader) -> None:
        """初始化算法训练和验证数据，需要在训练前调用

        Args:
            train_loader (_type_): 训练数据集加载器.
            val_loader (_type_): 验证数据集加载器.
        """
        self.train_loader = train_data_loader
        self.val_loader = val_data_loader

    @abstractmethod
    def _configure_optimizers(self):
        """
        配置优化器
        """
        pass

    @abstractmethod
    def _configure_schedulers(self):
        """
        配置学习率调度器
        """
        pass

    @abstractmethod
    def train_epoch(self, epoch) -> float:
        pass

    @abstractmethod
    def validate(self) -> float:
        pass

    @abstractmethod
    def save(self, epoch: int, is_best: bool = False) -> None:
        pass

    @abstractmethod
    def load(self, epoch: int, is_best: bool = False) -> None:
        pass
