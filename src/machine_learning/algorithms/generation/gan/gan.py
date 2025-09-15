from typing import Literal, Mapping, Any

import torch
from torch.utils.tensorboard import SummaryWriter

from machine_learning.networks import Generator, Discriminator
from machine_learning.algorithms.base import AlgorithmBase
from machine_learning.types.aliases import FilePath
from machine_learning.utils.img import plot_imgs


class GAN(AlgorithmBase):
    def __init__(
        self,
        cfg: FilePath | Mapping[str, Any],
        generator: Generator,
        discriminator: Discriminator,
        name: str | None = "gan",
        device: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        Implementation of GAN algorithm

        Args:
            cfg (YamlFilePath, Mapping[str, Any]): Configuration of the algorithm, it can be yaml file path or cfg map.
            generator (Generator): Generator net required by the GAN algorithm.
            discriminator (Discriminator): Discriminator net required by the GAN algorithm.
            name (str, optional): Name of the algorithm. Defaults to "gan".
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): Running device. Defaults to "auto"-automatic selection by algorithm.
        """
        super().__init__(cfg=cfg, name=name, device=device)

        self.generator = generator
        self.discriminator = discriminator
        self._add_net("generator", self.generator)
        self._add_net("discriminator", self.discriminator)

        # -------------------- the dim of prior ------------------------
        self.z_dim = self.generator.input_dim

    def _configure_optimizers(self) -> None:
        self.opt_cfg = self.cfg["optimizer"]

        if self.opt_cfg["type"] == "Adam":
            self._optimizers.update(
                {
                    "generator": torch.optim.Adam(
                        params=self.nets["generator"].parameters(),
                        lr=self.opt_cfg["g_learning_rate"],
                        betas=(self.opt_cfg["g_beta1"], self.opt_cfg["g_beta2"]),
                        eps=self.opt_cfg["g_eps"],
                        weight_decay=self.opt_cfg["g_weight_decay"],
                    )
                }
            )
            self._optimizers.update(
                {
                    "discriminator": torch.optim.Adam(
                        params=self.nets["discriminator"].parameters(),
                        lr=self.opt_cfg["d_learning_rate"],
                        betas=(self.opt_cfg["d_beta1"], self.opt_cfg["d_beta2"]),
                        eps=self.opt_cfg["d_eps"],
                        weight_decay=self.opt_cfg["d_weight_decay"],
                    )
                }
            )
        else:
            ValueError(f"Does not support optimizer:{self.opt_cfg['type']} currently.")

    def _configure_schedulers(self) -> None:
        sched_config = self.cfg.get("scheduler", {})

        if sched_config is not None and sched_config.get("type") == "StepLR":
            self._schedulers.update(
                {"generator": torch.optim.lr_scheduler.StepLR(self._optimizers["generator"], step_size=30, gamma=0.1)}
            )
            self._schedulers.update(
                {
                    "discriminator": torch.optim.lr_scheduler.StepLR(
                        self._optimizers["discriminator"], step_size=30, gamma=0.1
                    )
                }
            )

    def train_epoch(self, epoch: int, writer: SummaryWriter, log_interval: int = 10):
        """train a single epoch"""
        self.set_train()
        total_d_loss = 0.0
        total_g_loss = 0.0
        g_count = 0

        n_discriminator = self.cfg["algorithm"].get("n_discriminator", 1)

        for batch_idx, (real_images, _) in enumerate(self.train_loader):
            real_images = real_images.to(self._device, non_blocking=True)
            z = torch.randn((len(real_images), self.z_dim), device=self.device, dtype=torch.float32)
            fake_image = self.nets["generator"](z)

            # train discriminator
            self._optimizers["discriminator"].zero_grad()

            real_preds = self.nets["discriminator"](real_images)
            fake_preds = self.nets["discriminator"](fake_image.detach())

            d_loss = discriminator_criterion_bce(real_preds, fake_preds)
            d_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.nets["discriminator"].parameters(), self.cfg["optimizer"]["grad_clip"]["discriminator"]
            )
            self._optimizers["discriminator"].step()
            total_d_loss += d_loss

            real_accuracy = (real_preds > 0.5).float().mean()
            fake_accuracy = (fake_preds < 0.5).float().mean()

            # train generator
            if batch_idx % n_discriminator == 0:
                self._optimizers["generator"].zero_grad()

                fake_preds = self.nets["discriminator"](fake_image)
                g_loss = generator_criterion_bce(fake_preds)
                total_g_loss += g_loss

                g_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.nets["generator"].parameters(), self.cfg["optimizer"]["grad_clip"]["generator"]
                )
                self._optimizers["generator"].step()

                g_count += 1

            # log data
            if batch_idx % log_interval == 0:
                writer.add_scalar("d_loss/train_batch", d_loss.item(), epoch * len(self.train_loader) + batch_idx)
                writer.add_scalar("g_loss/train_batch", g_loss.item(), epoch * len(self.train_loader) + batch_idx)
                writer.add_scalar(
                    "real_accuracy/train_batch", real_accuracy.item(), epoch * len(self.train_loader) + batch_idx
                )
                writer.add_scalar(
                    "fake_accuracy/train_batch", fake_accuracy.item(), epoch * len(self.train_loader) + batch_idx
                )

        d_avg_loss = total_d_loss / len(self.train_loader)
        g_avg_loss = total_g_loss / g_count

        return {"discriminator loss": d_avg_loss, "generator loss": g_avg_loss}

    def validate(self) -> dict[str, float]:
        """Validate after a single train epoch"""
        self.set_eval()

        d_total_loss = 0.0
        g_total_loss = 0.0

        with torch.no_grad():
            for real_image, _ in self.val_loader:
                real_image = real_image.to(self.device, non_blocking=True)
                z = torch.randn((len(real_image), self.z_dim), device=self.device, dtype=torch.float32)
                fake_image = self.nets["generator"](z)

                real_preds = self.nets["discriminator"](real_image)
                fake_preds = self.nets["discriminator"](fake_image)

                d_loss = discriminator_criterion_bce(real_preds, fake_preds)
                d_total_loss += d_loss.item()

                g_loss = generator_criterion(fake_preds)
                g_total_loss += g_loss.item()

        d_avg_loss = d_total_loss / len(self.val_loader)
        g_avg_loss = g_total_loss / len(self.val_loader)

        return {"discriminator loss": d_avg_loss, "generator loss": g_avg_loss}

    def eval(self, num_samples: int = 5) -> None:
        """Evaluate the model effect by visualizing reconstruction results"""
        self.set_eval()

        z = torch.randn((num_samples, self.z_dim), device=self.device, dtype=torch.float32)

        with torch.no_grad():  # Disable gradient calculation, which has the same effect as.detach()
            recons = self.nets["generator"](z)

        plot_imgs(list(recons), color_mode="gray", backend="pyplot")


"""
Helper functions
"""


# loss fun1
def discriminator_criterion(real_preds: torch.Tensor, fake_preds: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    loss = -(torch.log(real_preds + eps).mean() + torch.log(1 - fake_preds + eps).mean())
    return loss


def generator_criterion(fake_preds: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    loss = -torch.log(fake_preds + eps).mean()
    return loss


# loss fun2
def discriminator_criterion_bce(real_preds: torch.Tensor, fake_preds: torch.Tensor) -> torch.Tensor:
    real_loss = torch.nn.functional.binary_cross_entropy(real_preds, torch.ones_like(real_preds))
    fake_loss = torch.nn.functional.binary_cross_entropy(fake_preds, torch.zeros_like(fake_preds))
    return real_loss + fake_loss


def generator_criterion_bce(fake_preds: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy(fake_preds, torch.ones_like(fake_preds))


# loss fun3
def discriminator_criterion_bce_logits(real_preds: torch.Tensor, fake_preds: torch.Tensor) -> torch.Tensor:
    real_loss = torch.nn.functional.binary_cross_entropy_with_logits(real_preds, torch.ones_like(real_preds))
    fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(fake_preds, torch.zeros_like(fake_preds))
    return real_loss + fake_loss


def generator_criterion_bce_logits(fake_preds: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(fake_preds, torch.ones_like(fake_preds))
