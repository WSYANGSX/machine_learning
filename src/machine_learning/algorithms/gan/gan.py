import os
import yaml
from tqdm import trange
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

torch.set_printoptions(threshold=torch.inf)


class GAN(nn.Module):
    def __init__(
        self,
        config_file: str,
        input_dim: int,
        generator: nn.Module,
        discriminator: nn.Module,
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        super().__init__()

        """
        生成对抗网络实现

        Args:
            config_file (str): 配置文件路径.
            input_dim (int): 生成器输入的维度.
            generator (nn.Module): 生成器定义.
            discriminator (nn.Module): 判别器定义.
            device (Literal["cuda", "cpu", "auto"], optional): 模型运行设备. Defaults to "auto".
        """

        # -------------------- 设备配置 ---------------------
        self.device = self._configure_device(device)

        # -------------------- 配置加载 ---------------------
        self.config = self._load_config(config_file)
        self._validate_config()

        # -------------------- 模型构建 ---------------------
        self.input_dim = input_dim
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        print("Model structure of generator and discriminator...")
        self.generator.view_structure()
        self.discriminator.view_structure()

        # -------------------- 权重初始化 -------------------
        if self.config["model"]["initialize_weights"]:
            self._initialize_weights()

        # -------------------- 配置优化器 -------------------
        self._configure_optimizer()
        self._configure_scheduler()

        # ----------------- 配置数据转换器 -------------------
        self._configure_transform()

        # -------------------- 数据加载 ---------------------
        self._load_datasets()

        # -------------------- 数据记录 ---------------------
        self.writer = SummaryWriter(log_dir=self.config["logging"]["log_dir"])
        self.best_loss = float("inf")

        # --------------------- 先验 -----------------------
        self.prior = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.input_dim, device=self.device),
            covariance_matrix=torch.eye(self.input_dim, device=self.device),
        )

    def _configure_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_config(self, config_file: str) -> dict:
        assert os.path.splitext(config_file)[1] == ".yaml" or os.path.splitext(config_file)[1] == ".yml", (
            "Please utilize a yaml configuration file."
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

    def _configure_optimizer(self) -> None:
        opt_config = self.config["optimizer"]

        if opt_config["type"] == "Adam":
            self.generator_optimizer = torch.optim.Adam(
                params=self.generator.parameters(),
                lr=opt_config["g_learning_rate"],
                betas=(opt_config["g_beta1"], opt_config["g_beta2"]),
                eps=opt_config["g_eps"],
                weight_decay=opt_config["g_weight_decay"],
            )
            self.discriminator_optimizer = torch.optim.Adam(
                params=self.discriminator.parameters(),
                lr=opt_config["d_learning_rate"],
                betas=(opt_config["d_beta1"], opt_config["d_beta2"]),
                eps=opt_config["d_eps"],
                weight_decay=opt_config["d_weight_decay"],
            )

        elif opt_config["type"] == "SGD":
            self.generator_optimizer = torch.optim.SGD(
                params=self.generator.parameters(),
                lr=opt_config["g_learning_rate"],
                momentum=opt_config["g_momentum"],
                dampening=opt_config["g_dampening"],
                weight_decay=opt_config["g_weight_decay"],
            )
            self.discriminator_optimizer = torch.optim.SGD(
                params=self.discriminator.parameters(),
                lr=opt_config["d_learning_rate"],
                momentum=opt_config["d_momentum"],
                dampening=opt_config["d_dampening"],
                weight_decay=opt_config["d_weight_decay"],
            )

        else:
            ValueError(f"暂时不支持优化器:{opt_config['type']}")

    def _configure_scheduler(self) -> None:
        self.generator_scheduler = None
        self.discriminator_scheduler = None

        sched_config = self.config["optimizer"].get("scheduler", {})
        if sched_config.get("type") == "ReduceLROnPlateau":
            self.generator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.generator_optimizer,
                mode="min",
                factor=sched_config.get("factor", 0.1),
                patience=sched_config.get("patience", 10),
            )
            self.discriminator_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.discriminator_optimizer,
                mode="min",
                factor=sched_config.get("factor", 0.1),
                patience=sched_config.get("patience", 10),
            )

    def _configure_transform(self) -> None:
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config["data"]["norm_mean"], std=self.config["data"]["norm_std"]),
            ]
        )

    def _initialize_weights(self) -> None:
        print("Initializing weights of encoder and decoder...")
        self.generator._initialize_weights()
        self.discriminator._initialize_weights()

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

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """保存模型检查点"""
        state = {
            "epoch": epoch,
            "generator_state": self.generator.state_dict(),
            "discriminator_state": self.discriminator.state_dict(),
            "generator_optimizer_state": self.generator_optimizer.state_dict(),
            "discriminator_optimizer_state": self.discriminator_optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
        }

        filename = f"checkpoint_epoch_{epoch}.pth"
        if is_best:
            filename = "best_model.pth"

        save_path = os.path.join(self.config["logging"]["model_dir"], filename)
        torch.save(state, save_path)
        print(f"Saved checkpoint to {save_path}")

    def train_discriminator(self, epoch: int) -> float:
        """训练单个discriminator epoch"""
        self.generator.eval()
        self.discriminator.train()

        total_loss = 0.0

        for _, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)

            z_prior = self.prior.sample((len(data),))
            data_ = self.generator(z_prior)
            self.discriminator_optimizer.zero_grad()

            output_t = self.discriminator(data)
            output_f = self.discriminator(data_)

            loss = discriminator_criterion(output_t, output_f)
            loss.backward()  # 反向传播计算各权重的梯度

            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config["training"]["grad_clip"])
            self.discriminator_optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)

        # 验证集测试
        val_avg_loss = self.val_discriminator()

        # 学习率调整
        if self.discriminator_scheduler is not None:
            if isinstance(self.discriminator_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.discriminator_scheduler.step(val_avg_loss)
            else:
                self.discriminator_scheduler.step()

        # 打印日志
        if epoch % self.config["logging"].get("discriminator_log_interval", 10) == 0:
            print(
                f"Epoch: {epoch + 1:03d} | "
                f"Discriminator Loss: {avg_loss:.4f} | "
                f"Discriminator Val Loss: {val_avg_loss:.4f} | "
                f"LR: {self.discriminator_scheduler.get_last_lr()[0]:.2e}"
            )

        return avg_loss

    def val_discriminator(self) -> float:
        self.discriminator.eval()
        val_total_loss = 0.0

        with torch.no_grad():
            for data, _ in self.validate_loader:
                data = data.to(self.device, non_blocking=True)

                z_prior = self.prior.sample((len(data),))
                data_ = self.generator(z_prior)

                output_t = self.discriminator(data)
                output_f = self.discriminator(data_)

                loss = discriminator_criterion(output_t, output_f)
                val_total_loss += loss.item()

        return val_total_loss / len(self.validate_loader)

    def train_generator(self, epoch: int):
        """训练单个generator epoch"""
        self.generator.train()
        self.discriminator.eval()

        total_loss = 0.0

        for _, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)

            z_prior = self.prior.sample((len(data),))
            data_ = self.generator(z_prior)

            self.generator_optimizer.zero_grad()

            output_f = self.discriminator(data_)

            loss = generator_criterion(output_f)
            loss.backward()  # 反向传播计算各权重的梯度

            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config["training"]["grad_clip"])
            self.generator_optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)

        # 验证集测试
        val_avg_loss = self.val_generator()

        # 学习率调整
        if self.generator_scheduler is not None:
            if isinstance(self.generator_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.generator_scheduler.step(val_avg_loss)
            else:
                self.generator_scheduler.step()

        # 打印日志
        if epoch % self.config["logging"].get("generator_log_interval", 10) == 0:
            print(
                f"Epoch {epoch + 1:03d} | "
                f"Generator Loss: {avg_loss:.4f} | "
                f"Generator Val Loss: {val_avg_loss:.4f} | "
                f"LR: {self.generator_scheduler.get_last_lr()[0]:.2e}"
            )

        return avg_loss

    def val_generator(self) -> float:
        self.generator.eval()
        val_total_loss = 0.0

        with torch.no_grad():
            for data, _ in self.validate_loader:
                data = data.to(self.device, non_blocking=True)

                z_prior = self.prior.sample((len(data),))
                data_ = self.generator(z_prior)

                output_f = self.discriminator(data_)

                loss = generator_criterion(output_f)
                val_total_loss += loss.item()

        return val_total_loss / len(self.validate_loader)

    def train_model(self) -> None:
        """完整训练"""
        print("Start training...")
        for epoch in trange(self.config["training"]["epochs"]):
            # 每个epoch进行多次判别器训练
            for _ in range(self.config["training"].get("n_discriminator", 5)):
                d_loss = self.train_discriminator(epoch)
            # 训练生成器
            g_loss = self.train_generator(epoch)

            # 记录验证损失
            self.writer.add_scalar("D_Loss/epoch", d_loss, epoch)
            self.writer.add_scalar("G_Loss/epoch", g_loss, epoch)

            # # 保存最佳模型
            # if g_loss < self.best_loss:
            #     self.best_loss = g_loss
            #     self.save_checkpoint(epoch, is_best=True)

            # # 定期保存
            # if (epoch + 1) % self.config["training"]["save_interval"] == 0:
            #     self.save_checkpoint(epoch)

            # 打印日志
            print(
                f"Epoch: {epoch + 1:03d} | "
                f"Discriminator Loss: {d_loss:.4f} | "
                f"Generator Loss: {g_loss:.4f} | "
                f"D_LR: {self.discriminator_optimizer.param_groups[0]['lr']:.2e} | "
                f"G_LR: {self.generator_optimizer.param_groups[0]['lr']:.2e}"
            )

    def visualize_reconstruction(self, num_samples: int = 5) -> None:
        """可视化重构结果"""
        self.generator.eval()
        self.discriminator.eval()

        z = self.prior.sample((num_samples,))

        with torch.no_grad():
            reconstructions = self.generator(z)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        for i in range(num_samples):
            # 重构图像
            ax = plt.subplot(1, num_samples, i + 1)
            plt.imshow(reconstructions[i].cpu().squeeze(), cmap="gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


"""
Helper functions
"""


def discriminator_criterion(real_preds: torch.Tensor, fake_preds: torch.Tensor) -> float:
    real_loss = torch.nn.functional.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
    fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(fake_preds, torch.zeros_like(fake_preds))
    return real_loss + fake_loss


def generator_criterion(fake_preds: torch.Tensor) -> float:
    return torch.nn.functional.binary_cross_entropy_with_logits(fake_preds, torch.ones_like(fake_preds))
