import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 超参数设置
batch_size = 128
latent_dim = 100
lr = 0.0002
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 将像素值归一化到[-1, 1]
    ]
)

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 生成器定义
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),  # 输出值在[-1, 1]之间
        )

    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)


# 判别器定义
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.main(x)


# 初始化模型
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 损失函数和优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 训练过程记录
G_losses = []
D_losses = []

# 训练循环
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # 真实标签和假标签
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # =============================
        #  训练判别器
        # =============================
        optimizer_D.zero_grad()

        # 真实图像的损失
        outputs_real = discriminator(real_images)
        loss_real = criterion(outputs_real, real_labels)

        # 生成假图像
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)

        # 假图像的损失
        outputs_fake = discriminator(fake_images.detach())
        loss_fake = criterion(outputs_fake, fake_labels)

        # 总损失和反向传播
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # =============================
        #  训练生成器
        # =============================
        optimizer_G.zero_grad()

        # 生成器试图欺骗判别器
        outputs = discriminator(fake_images)
        loss_G = criterion(outputs, real_labels)

        loss_G.backward()
        optimizer_G.step()

        # 记录损失
        if i % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(train_loader)} "
                f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}"
            )

        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

    # 每个epoch结束时生成示例图像
    with torch.no_grad():
        test_z = torch.randn(16, latent_dim).to(device)
        generated = generator(test_z).cpu().detach()

        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Generated Images at Epoch {epoch}")
        plt.imshow(np.transpose(torchvision.utils.make_grid(generated, nrow=4, padding=2, normalize=True), (1, 2, 0)))
        plt.show()

# 绘制训练曲线
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
