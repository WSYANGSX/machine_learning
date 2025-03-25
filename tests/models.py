def train_discriminator(self) -> float:
        """训练单个discriminator epoch"""
        self.models["generator"].eval()
        self.models["discriminator"].train()

        total_loss = 0.0

        for _, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)

            z_prior = torch.randn((len(data), self.z_dim), device=self.device, dtype=torch.float32)
            data_ = self.models["generator"](z_prior)

            self._optimizers["discriminator"].zero_grad()

            real_preds = self.models["discriminator"](data)
            fake_preds = self.models["discriminator"](data_)

            loss = discriminator_criterion(real_preds, fake_preds)
            loss.backward()  # 反向传播计算各权重的梯度

            torch.nn.utils.clip_grad_norm_(
                self.models["discriminator"].parameters(), self.cfg["training"]["grad_clip"]["discriminator"]
            )
            self._optimizers["discriminator"].step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)

        return avg_loss

    def eval_discriminator(self) -> float:
        self.models["discriminator"].eval()
        self.models["generator"].eval()

        val_total_loss = 0.0

        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device, non_blocking=True)

                z_prior = torch.randn((len(data), self.z_dim), device=self.device, dtype=torch.float32)
                data_ = self.models["generator"](z_prior)

                real_preds = self.models["discriminator"](data)
                fake_preds = self.models["discriminator"](data_)

                loss = discriminator_criterion(real_preds, fake_preds)
                val_total_loss += loss.item()

        avg_loss = val_total_loss / len(self.val_loader)

        return avg_loss

    def train_generator(self):
        """训练单个generator epoch"""
        self.models["generator"].train()
        self.models["discriminator"].eval()

        total_loss = 0.0

        for _, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device, non_blocking=True)

            z_prior = torch.randn((len(data), self.z_dim), device=self.device, dtype=torch.float32)
            data_ = self.models["generator"](z_prior)

            self._optimizers["generator"].zero_grad()

            fake_preds = self.models["discriminator"](data_)

            loss = generator_criterion(fake_preds)
            loss.backward()  # 反向传播计算各权重的梯度

            torch.nn.utils.clip_grad_norm_(
                self.models["generator"].parameters(), self.cfg["training"]["grad_clip"]["generator"]
            )
            self._optimizers["generator"].step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)

        return avg_loss

    def eval_generator(self) -> float:
        self.models["generator"].eval()
        self.models["discriminator"].eval()

        val_total_loss = 0.0

        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device, non_blocking=True)

                z_prior = torch.randn((len(data), self.z_dim), device=self.device, dtype=torch.float32)
                data_ = self.models["generator"](z_prior)

                fake_preds = self.models["discriminator"](data_)

                loss = generator_criterion(fake_preds)
                val_total_loss += loss.item()

        avg_loss = val_total_loss / len(self.val_loader)

        return avg_loss