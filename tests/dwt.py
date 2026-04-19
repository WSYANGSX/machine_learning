import torch
import torch.nn as nn
import pywt
import ptwt


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

        # ==========================================
        # 2. 计算收益：精准抛弃没用的对角线噪声 (HH)
        # ==========================================
        # 将对角线高频 hh 置零。使用 torch.zeros_like 保持设备和梯度追踪一致
        hh_dropped = torch.zeros_like(hh)

        # ==========================================
        # 3. 还原过程：逆小波变换 (IDWT)
        # ==========================================
        # 将干预后的系数重新打包，准备重构
        enhanced_coeffs = [ll_enhanced, (hl_enhanced, lh_enhanced, hh_dropped)]

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
    x = torch.randn(2, 64, 32, 32, requires_grad=True).to(device)

    model = DWTFeatureEnhancer(wavelet_name="haar").to(device)

    # 纯测试：完全无损的数学重构验证 (不丢弃 HH)
    coeffs_test = ptwt.wavedec2(x, pywt.Wavelet("haar"), level=1)
    x_perfect = ptwt.waverec2(coeffs_test, pywt.Wavelet("haar"))
    max_error = torch.max(torch.abs(x - x_perfect))
    print(f"100% 完美数学重构的最大误差: {max_error.item():.8f} (接近于0表示完全无损)")

    # 网络实际前向传播测试 (丢弃了 HH 噪声)
    out = model(x)
    print(f"原始特征图维度: {x.shape}")
    print(f"处理后特征图维度: {out.shape}")

    # 测试反向传播是否正常通畅
    loss = out.sum()
    loss.backward()
    print("梯度形状是否与输入对齐:", x.grad.shape == x.shape)
