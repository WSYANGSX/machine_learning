import torch
import torch.nn as nn
import pywt
import ptwt
from PIL import Image
import numpy as np

from machine_learning.utils.ops import img_np2tensor, img_tensor2np
from machine_learning.utils.plots import plot_imgs


class DWTFeatureEnhancer(nn.Module):
    """
    基于离散小波变换 (DWT) 的特征降维与增强模块
    """

    def __init__(self, wavelet_name="haar"):
        super().__init__()
        # 初始化小波基，'haar' 是最常用且计算最快的小波基
        self.wavelet = pywt.Wavelet(wavelet_name)

    def forward(self, x):
        """
        x: 输入张量，形状为 (B, C, H, W)
        """
        # ==========================================
        # 1. 降维过程（分解）：2D DWT
        # ==========================================
        # level=1 表示进行一次分解，H 和 W 会缩小为原来的一半
        coeffs = ptwt.wavedec2(x, self.wavelet, level=1)

        # 解析返回的系数
        # ll: 低频核心结构 (B, C, H/2, W/2)
        # hl: 水平高频 (B, C, H/2, W/2)
        # lh: 垂直高频 (B, C, H/2, W/2)
        # hh: 对角线高频 (B, C, H/2, W/2)
        ll, (hl, lh, hh) = coeffs

        # ------------------------------------------
        # 在这里，你可以在 16x16 的分辨率上大显身手！
        # 例如：对 ll 应用注意力机制，或者对 hl, lh 提取边缘特征
        # ll_enhanced = self.attention(ll)
        # hl_enhanced = self.edge_conv(hl)
        # lh_enhanced = self.edge_conv(lh)
        # ------------------------------------------

        # 假设我们只做原样传递，保留核心结构和水平/垂直边缘
        ll_enhanced = ll
        hl_enhanced = hl
        lh_enhanced = lh
        hh_enhanced = hh

        # ==========================================
        # 2. 计算收益：精准抛弃没用的对角线噪声 (HH)
        # ==========================================
        # 将对角线高频 hh 置零。使用 torch.zeros_like 保持设备和梯度追踪一致
        # ll_enhanced = torch.zeros_like(ll)
        hl_enhanced = torch.zeros_like(hl)
        lh_enhanced = torch.zeros_like(lh)
        hh_enhanced = torch.zeros_like(hh)

        # ==========================================
        # 3. 还原过程：逆小波变换 (IDWT)
        # ==========================================
        # 将干预后的系数重新打包，准备重构
        enhanced_coeffs = [ll_enhanced, (hl_enhanced, lh_enhanced, hh_enhanced)]

        # 无损拼合还原回 32x32 分辨率
        x_reconstructed = ptwt.waverec2(enhanced_coeffs, self.wavelet)

        return x_reconstructed


# ==========================================
# 4. 测试与验证代码
# ==========================================
if __name__ == "__main__":
    # 模拟输入: Batch=2, Channels=64, H=32, W=32
    # 确保 tensor 在 GPU 上以测试 CUDA 加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img1 = np.array(Image.open("/home/yangxf/WorkSpace/datasets/..datasets/car/imgs/test/3f8d611822bc_02.jpg"))

    img1 = img_np2tensor(img1).detach().clone().to(device).requires_grad_(True)

    model = DWTFeatureEnhancer(wavelet_name="haar").to(device)

    # 纯测试：完全无损的数学重构验证 (不丢弃 HH)
    coeffs_test = ptwt.wavedec2(img1, pywt.Wavelet("haar"), level=1)
    x_perfect = ptwt.waverec2(coeffs_test, pywt.Wavelet("haar"))
    max_error = torch.max(torch.abs(img1 - x_perfect))
    print(f"100% 完美数学重构的最大误差: {max_error.item():.8f} (接近于0表示完全无损)")

    # 网络实际前向传播测试 (丢弃了 HH 噪声)
    out = model(img1)
    print(f"原始特征图维度: {img1.shape}")
    print(f"处理后特征图维度: {out.shape}")

    # 有无损的数学重构验证 (不丢弃 HH)
    max_error2 = torch.max(torch.abs(img1 - out.clamp(0, 1)))
    print(f"100% 完美数学重构的最大误差: {max_error2.item():.8f} (接近于0表示完全无损)")
    plot_imgs([img_tensor2np(out.clamp(0, 1))])

    # 测试反向传播是否正常通畅
    loss = out.sum()
    loss.backward()
    print("梯度形状是否与输入对齐:", img1.grad.shape == img1.shape)
