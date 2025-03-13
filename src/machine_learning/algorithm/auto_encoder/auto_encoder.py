# 对于无监督式学习，比较好的办法是重建自己，通过重建数据发现数据的模态特征信息
# auto-encoder相当于对数据进行降维处理，类似PCA，只不过PCA是通过求解特征向量进行降维，是线性降维方式，而auto-encoder是非线性降维方式
import os
import yaml
from tqdm import trange
from typing import Literal
from collections import OrderedDict
from torch.utils.data import DataLoader
from src.machine_learning.utils.data_utils import CustomDataset, data_parse

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


class AutoEncoder(nn.Module):
    def __init__(
        self,
        config_file: str,
        encoder_layers: OrderedDict[nn.Module],
        decoder_layers: OrderedDict[nn.Module],
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        自编码器实现

        parameters:
        - config_file: 配置文件路径(YAML格式)
        - encoder_layers: 编码器层定义(OrderedDict)
        - decoder_layers: 解码器层定义(OrderedDict)
        - device: 运行设备(auto自动选择)
        """
        super().__init__()

        # -------------------- 设备配置 --------------------
        self.device = self._configure_device(device)

        # -------------------- 配置加载 --------------------
        self.config = self._load_config(config_file)
        self._validate_config()

        # -------------------- 模型构建 --------------------
        encoder = self._build_module(encoder_layers, "Encoder")
        decoder = self._build_module(decoder_layers, "Decoder")
        self.model = nn.Sequential(encoder, decoder).to(self.device)

        # -------------------- 权重初始化 --------------------
        if self.config["model"]["initialize_weights"]:
            self._initialize_weights()

        # -------------------- 配置优化器 --------------------
        self._configure_optimizer()
        self._configure_scheduler()

        # ----------------- 配置数据转换器 -------------------
        self._configure_transform()

        # -------------------- 数据加载 --------------------
        self._load_datasets()

        # -------------------- 数据记录 --------------------
        self.writer = SummaryWriter(log_dir=self.config["logging"]["log_dir"])
        self.best_loss = float("inf")

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
        print("configuration parameters: ", config)

        return config

    def _validate_config(self):
        """配置参数验证"""
        required_sections = ["data", "model", "training", "optimizer", "logging"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件中缺少必要部分: {section}")

    def _configure_optimizer(self) -> None:
        opt_config = self.config["optimizer"]
        if opt_config["type"] == "Adam":
            self.optimizer = torch.optim.Adam(
                params=self.model.parameters(),
                lr=opt_config["learning_rate"],
                betas=(opt_config["beta1"], opt_config["beta2"]),
                eps=opt_config["eps"],
                weight_decay=opt_config["weight_decay"],
            )
        elif opt_config["type"] == "SGD":
            self.optimizer = torch.optim.SGD(
                params=self.model.parameters(),
                lr=opt_config["learning_rate"],
                momentum=opt_config["momentum"],
                dampening=opt_config["dampening"],
                weight_decay=opt_config["weight_decay"],
            )
        else:
            ValueError(f"暂时不支持优化器:{opt_config['type']}")

    def _configure_scheduler(self) -> None:
        self.scheduler = None
        sched_config = self.config["optimizer"].get("scheduler", {})
        if sched_config.get("type") == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=sched_config.get("factor", 0.1),
                patience=sched_config.get("patience", 10),
                verbose=True,
            )

    def _configure_transform(self) -> None:
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config["data"]["norm_mean"], std=self.config["data"]["norm_std"]),
            ]
        )

    def _initialize_weights(self) -> None:
        print("Initializing weights with Kaiming normal...")
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _build_module(self, layers: OrderedDict[nn.Module], module_name: str) -> nn.Sequential:
        print(f"Building {module_name}")
        for name, layer in layers.items():
            print(f"  - {name}: {layer}")
        return nn.Sequential(layers)

    def _load_datasets(self) -> None:
        print("Loading datasets...")
        train_data, train_labels, validate_data, validate_labels = data_parse(self.config["data"]["data_path"])

        # 创建dataset和datasetloader
        train_dataset = CustomDataset(train_data, train_labels, self.transform)
        validate_dataset = CustomDataset(validate_data, validate_labels, self.transform)

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

    def train_epoch(self, epoch: int) -> float:
        """训练单个epoch"""
        self.model.train()
        total_loss = 0.0
        criterion = nn.MSELoss()

        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, data)
            loss.backward()  # 反向传播计算各权重的梯度

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["grad_clip"])
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % self.config["logging"].get("log_interval", 10) == 0:
                self.writer.add_scalar("Loss/train_batch", loss.item(), epoch * len(self.train_loader))  # batch loss

        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/train", avg_loss, epoch)  # epoch loss
        return avg_loss

    def validate(self) -> float:
        """验证步骤"""
        self.model.eval()
        total_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for data, _ in self.validate_loader:
                data = data.to(self.device, non_blocking=True)
                recon = self.model(data)
                total_loss += criterion(recon, data).item()

        avg_loss = total_loss / len(self.validate_loader)
        return avg_loss

    def train_model(self) -> None:
        """完整训练"""
        print("Start training...")
        for epoch in trange(self.config["training"]["epochs"]):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            # 学习率调整
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 记录验证损失
            self.writer.add_scalar("Loss/val", val_loss, epoch)

            # 保存最佳模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

            # 定期保存
            if (epoch + 1) % self.config["training"]["save_interval"] == 0:
                self.save_checkpoint(epoch)

            # 打印日志
            print(
                f"Epoch {epoch + 1:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """保存模型检查点"""
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
        }

        filename = f"checkpoint_epoch_{epoch}.pth"
        if is_best:
            filename = "best_model.pth"

        save_path = os.path.join(self.config["logging"]["model_dir"], filename)
        torch.save(state, save_path)
        print(f"Saved checkpoint to {save_path}")

    def visualize_reconstruction(self, num_samples: int = 5) -> None:
        """可视化重构结果"""
        self.model.eval()

        data, _ = next(iter(self.validate_loader))
        sample_indices = torch.randint(low=0, high=len(data), size=(num_samples,))
        data = data[sample_indices].to(self.device)

        with torch.no_grad():
            reconstructions = self.model(data)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        for i in range(num_samples):
            # 原始图像
            ax = plt.subplot(2, num_samples, i + 1)
            plt.imshow(data[i].cpu().squeeze(), cmap="gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # 重构图像
            ax = plt.subplot(2, num_samples, i + 1 + num_samples)
            plt.imshow(reconstructions[i].cpu().squeeze(), cmap="gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
