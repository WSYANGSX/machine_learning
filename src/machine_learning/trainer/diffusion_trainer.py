from .trainer import Trainer


class DiffusionTrainer(Trainer):
    def __init__(self, cfg, data, transform, algo):
        super().__init__(cfg, data, transform, algo)

    def train(self, start_epoch=0):
        """完整训练"""
        print("[INFO] Start training...")

        # 首先进行前向过程的计算

        for epoch in trange(start_epoch, self.cfg.get("epochs", 100)):
            train_loss = self._algorithm.train_epoch(epoch, self.writer, self.cfg.get("log_interval", 10))
            val_loss = self._algorithm.validate()

            # 学习率调整
            if self._algorithm._schedulers:
                for key, val in self._algorithm._schedulers.items():
                    if isinstance(val, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        val.step(val_loss[key])
                    else:
                        val.step()

            # 记录训练损失
            for key, val in train_loss.items():
                self.writer.add_scalar(f"{key} loss/train", val, epoch)

            # 记录验证损失
            for key, val in val_loss.items():
                self.writer.add_scalar(f"{key} loss/val", val, epoch)

            # 保存最佳模型
            if "save_metric" in val_loss:
                if val_loss["save_metric"] < self.best_loss:
                    self.best_loss = val_loss["save_metric"]
                    self.save_checkpoint(epoch, val_loss, self.best_loss, is_best=True)
            else:
                print("Val loss has no save metric, saving of the best loss model skipped.")

            # 定期保存
            if (epoch + 1) % self.cfg.get("save_interval", 10) == 0:
                self.save_checkpoint(epoch, val_loss, self.best_loss, is_best=False)

            # 打印日志
            print(f"Epoch: {epoch + 1:03d} | ", end="")
            for key, val in train_loss.items():
                print(f"{key} train loss {val:.4f} | ", end="")
            for key, val in val_loss.items():
                print(f"{key} val loss {val:.4f} | ", end="")
            for key, opt in self._algorithm._optimizers.items():
                print(f"{key} lr: {opt.param_groups[0]['lr']:.2e} | ")
