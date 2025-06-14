import os
import yaml
from typing import Literal, Mapping, Any
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from machine_learning.models import BaseNet
from machine_learning.utils import print_dict, print_segmentation


class AlgorithmBase(ABC):
    def __init__(
        self,
        cfg: str | dict,
        models: Mapping[str, BaseNet],
        name: str | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ):
        """算法抽象基类

        Args:
            cfg (str): 算法配置, YAML文件路径或者配置字典.
            models (Mapping[str, BaseNet]): 算法所需的网络模型.
            name (str | None, optional): 算法名称. Defaults to None.
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): 算法运行设备. Defaults to "auto".
        """
        super().__init__()

        self._models = models
        self._optimizers = {}
        self._schedulers = {}

        # -------------------- 设备配置 --------------------
        self._device = self._configure_device(device)

        # -------------------- 配置加载 --------------------
        self._cfg = self._load_config(cfg)
        self._validate_config()

        # -------------------- 设备配置 --------------------
        self._name = name if name is not None else self._cfg.get("algorithm", {}).get("name", __class__.__name__)

        # -------------------- 配置模型 --------------------
        self._configure_models()
        if self.cfg["model"]["initialize_weights"]:
            self._initialize_weights()

    @property
    def name(self) -> str:
        return self._name

    @property
    def models(self) -> dict[str, BaseNet]:
        return self._models

    @property
    def optimizers(self) -> dict[str, BaseNet]:
        return self._optimizers

    @property
    def schedulers(self) -> dict[str, BaseNet]:
        return self._schedulers

    @property
    def cfg(self) -> dict:
        return self._cfg

    @property
    def device(self) -> torch.device:
        return self._device

    def _configure_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_config(self, config: str | dict) -> dict:
        if isinstance(config, dict):
            cfg = config
        else:
            assert os.path.splitext(config)[1] == ".yaml" or os.path.splitext(config)[1] == ".yml", (
                "Please ultilize a yaml configuration file."
            )
            with open(config, "r") as f:
                cfg = yaml.safe_load(f)

        print_segmentation()
        print("Configuration parameters: ")
        print_dict(cfg)
        print_segmentation()

        return cfg

    def _validate_config(self):
        """配置参数验证"""
        required_sections = ["algorithm", "model", "optimizer", "scheduler"]
        for section in required_sections:
            if section not in self.cfg:
                raise ValueError(f"配置文件中缺少必要部分: {section}")

    def _configure_models(self):
        """
        配置模型
        """
        for model in self._models.values():
            model.to(self._device)
            model.view_structure()

    def _initialize_weights(self) -> None:
        for model in self._models.values():
            model._initialize_weights()

    def _initialize_dependent_on_data(
        self, train_loader: DataLoader, val_loader: DataLoader, batch_size: int, **kwargs
    ) -> None:
        """初始化算法训练和验证数据加载器和其他需要数据解析后才能确定的参数，需要在训练前调用

        Args:
            train_loader (_type_): 训练数据集加载器.
            val_loader (_type_): 验证数据集加载器.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size

        # 保护关键属性不被覆盖
        protected_attrs = {"train_loader", "val_loader", "batch_size"}

        for key, val in kwargs.items():
            if key in protected_attrs:
                print(f"Attempted to override protected attribute '{key}'. Ignored.")
                continue

            if hasattr(self, key):
                # 对已有属性使用 setattr 确保描述符正常工作
                setattr(self, key, val)
                print(f"[INFO] Set {key} attribute of {self.__class__.__name__} to new value.")
            else:
                setattr(self, key, val)
                print(f"[INFO] {self.__class__.__name__.capitalize()} set new attribute: {key}")

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
    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int) -> dict[str, float]:
        pass

    @abstractmethod
    def validate(self) -> dict[str, float]:
        pass

    @abstractmethod
    def eval(self, num_samples: int) -> None:
        pass

    def save(self, epoch: int, loss: dict, best_loss: float, save_path: str) -> None:
        """保存模型检查点"""
        state = {"epoch": epoch, "cfg": self.cfg, "loss": loss, "best loss": best_loss, "models": {}, "optimizers": {}}

        # 保存模型参数
        for key, val in self.models.items():
            state["models"].update({key: val.state_dict()})

        # 保存优化器参数
        for key, val in self._optimizers.items():
            state["optimizers"].update({key: val.state_dict()})

        torch.save(state, save_path)
        print(f"Saved checkpoint to {save_path}")

    def load(self, checkpoint: str) -> tuple[Any]:
        state = torch.load(checkpoint)

        # 加载模型参数
        for key, val in self.models.items():
            val.load_state_dict(state["models"][key])

        # 加载优化器参数
        for key, val in self._optimizers.items():
            val.load_state_dict(state["optimizers"][key])

        epoch = state["epoch"]
        cfg = state["cfg"]
        loss = state["loss"]
        best_loss = state["best loss"]

        return {"epoch": epoch, "cfg": cfg, "loss": loss, "best_loss": best_loss}

    def set_train(self) -> None:
        """将算法中的所有模型设置为train model"""
        for model in self.models.values():
            model.train()

    def set_eval(self) -> None:
        """将算法中的所有模型设置为eval model"""
        for model in self.models.values():
            model.eval()
