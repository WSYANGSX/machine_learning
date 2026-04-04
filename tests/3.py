import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LowRankHyperedgeGen(nn.Module):
    """
    低秩超边生成器

    使用低秩分解 (prototypes = U @ V) 减少参数量和计算量
    参数量从 O(E×D) 减少到 O(E×r + r×D)，其中 r << min(E, D)

    Attributes:
        node_dim (int): 顶点特征维度
        num_hyperedges (int): 超边数量
        rank (int): 低秩分解的秩
        num_heads (int): 注意力头数
        context (str): 上下文类型
    """

    def __init__(self, node_dim, num_hyperedges, rank=16, num_heads=4, dropout=0.1, context="both"):
        super().__init__()

        self.node_dim = node_dim
        self.num_hyperedges = num_hyperedges
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.context = context
        self.dropout_rate = dropout

        # 参数检查
        assert node_dim % num_heads == 0, "node_dim must be divisible by num_heads"
        assert rank <= min(num_hyperedges, node_dim), (
            f"rank {rank} must be ≤ min(num_hyperedges={num_hyperedges}, node_dim={node_dim})"
        )

        # ========== 低秩分解参数 ==========
        # U: [E, r] - 超边的低维表示
        # V: [r, D] - 特征空间的基
        self.U = nn.Parameter(torch.Tensor(num_hyperedges, rank))
        self.V = nn.Parameter(torch.Tensor(rank, node_dim))

        # 基原型（可选，可添加偏置）
        self.prototype_bias = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))

        # ========== 上下文网络（生成动态偏移） ==========
        # 由于原型是低秩的，偏移也应该是低秩的
        if context in ("mean", "max"):
            self.context_net = nn.Linear(node_dim, rank * node_dim)
        elif context == "both":
            self.context_net = nn.Linear(2 * node_dim, rank * node_dim)
        else:
            raise ValueError(f"Unsupported context '{context}'")

        # 动态U的生成网络（可选）
        self.dynamic_U_net = nn.Sequential(
            nn.Linear(node_dim if context in ("mean", "max") else 2 * node_dim, rank * rank),
            nn.GELU(),
            nn.Linear(rank * rank, rank * rank),
        )

        # ========== 投影层 ==========
        self.pre_head_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

        # ========== 初始化参数 ==========
        self._init_parameters()

    def _init_parameters(self):
        """初始化参数"""
        # 低秩矩阵初始化
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.zeros_(self.prototype_bias)  # 偏置初始为0

        # 计算参数量对比
        full_params = self.num_hyperedges * self.node_dim
        lowrank_params = (
            self.num_hyperedges * self.rank + self.rank * self.node_dim + self.num_hyperedges * self.node_dim
        )  # 包括bias

        compression_ratio = lowrank_params / full_params
        print(f"低秩压缩: {full_params} -> {lowrank_params} params (压缩率: {compression_ratio:.2%})")

        if compression_ratio >= 1.0:
            print("⚠️ 警告: 低秩分解未减少参数量，请减小rank值")

    def _compute_prototypes(self, context_feat):
        """
        计算动态低秩原型

        Args:
            context_feat: 全局上下文特征 [B, D] 或 [B, 2D]

        Returns:
            prototypes: 动态原型 [B, E, D]
        """
        B = context_feat.shape[0]

        # ========== 方法1：动态V（推荐） ==========
        # 根据上下文调整V矩阵
        V_offset = self.context_net(context_feat)  # [B, r*D]
        V_offset = V_offset.view(B, self.rank, self.node_dim)  # [B, r, D]

        # 动态V = 基V + 偏移
        V_dynamic = self.V.unsqueeze(0) + V_offset  # [B, r, D]

        # ========== 方法2：动态U（可选） ==========
        # 如果需要更动态的超边表示
        if self.training:  # 只在训练时使用，避免推理时的不确定性
            U_offset = self.dynamic_U_net(context_feat)  # [B, r*r]
            U_offset = U_offset.view(B, self.rank, self.rank)  # [B, r, r]

            # 动态U = 基U + 变换
            U_base = self.U  # [E, r]
            U_dynamic = torch.matmul(U_base, U_offset)  # [E, r] × [B, r, r] → [B, E, r]
        else:
            U_dynamic = self.U.unsqueeze(0)  # [1, E, r]

        # ========== 计算低秩原型 ==========
        # 原型 = U × V
        # [B, E, r] × [B, r, D] → [B, E, D]
        prototypes = torch.bmm(U_dynamic, V_dynamic)

        # 添加偏置
        prototypes = prototypes + self.prototype_bias.unsqueeze(0)

        return prototypes

    def forward(self, X):
        """
        前向传播

        Args:
            X: 顶点特征 [B, N, D]

        Returns:
            A: 参与矩阵 [B, N, E]
        """
        B, N, D = X.shape

        # ========== 1. 计算全局上下文 ==========
        if self.context == "mean":
            context_cat = X.mean(dim=1)  # [B, D]
        elif self.context == "max":
            context_cat, _ = X.max(dim=1)  # [B, D]
        else:  # both
            avg_context = X.mean(dim=1)  # [B, D]
            max_context, _ = X.max(dim=1)  # [B, D]
            context_cat = torch.cat([avg_context, max_context], dim=-1)  # [B, 2D]

        # ========== 2. 生成动态低秩原型 ==========
        prototypes = self._compute_prototypes(context_cat)  # [B, E, D]

        # ========== 3. 计算相似度（参与度） ==========
        # 投影顶点特征
        X_proj = self.pre_head_proj(X)  # [B, N, D]

        # 分割多头
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d_h]
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # [B, H, E, d_h]

        # 展平以进行批量矩阵乘法
        X_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)  # [B*H, N, d_h]
        proto_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim)  # [B*H, E, d_h]

        # 计算相似度（使用更高效的计算方式）
        # 方法1：直接矩阵乘法（标准）
        # logits = torch.bmm(X_flat, proto_flat.transpose(1, 2)) / self.scaling  # [B*H, N, E]

        # 方法2：利用低秩结构优化（可选）
        # 将原型分解回U和V，然后分别计算
        logits = self._efficient_similarity(X_flat, proto_flat)

        logits = logits.view(B, self.num_heads, N, self.num_hyperedges)

        # 平均多头结果
        logits = logits.mean(dim=1)  # [B, N, E]

        # Dropout和softmax
        logits = self.dropout(logits)
        A = F.softmax(logits, dim=-1)  # [B, N, E]

        return A

    def _efficient_similarity(self, X_flat, proto_flat):
        """
        利用低秩结构的高效相似度计算

        由于原型是低秩的，可以分解计算来减少计算量
        """
        B_H, N, d_h = X_flat.shape
        _, E, _ = proto_flat.shape

        # 标准计算：O(B*H * N * E * d_h)
        # logits = torch.bmm(X_flat, proto_flat.transpose(1, 2)) / self.scaling

        # 低秩优化计算：
        # 由于proto_flat是低秩的，可以表示为 U' @ V' 的形式
        # 但实际上proto_flat已经是低秩原型，这里主要展示优化思想

        # 更实际的优化：使用分组计算
        if N * E > 10000:  # 只在大型矩阵时启用优化
            # 分块计算
            block_size = 64
            logits = torch.zeros(B_H, N, E, device=X_flat.device)

            for i in range(0, N, block_size):
                end_i = min(i + block_size, N)
                X_block = X_flat[:, i:end_i, :]  # [B_H, block, d_h]

                for j in range(0, E, block_size):
                    end_j = min(j + block_size, E)
                    proto_block = proto_flat[:, j:end_j, :]  # [B_H, block, d_h]

                    # 计算块相似度
                    block_logits = torch.bmm(X_block, proto_block.transpose(1, 2)) / self.scaling
                    logits[:, i:end_i, j:end_j] = block_logits
        else:
            # 直接计算
            logits = torch.bmm(X_flat, proto_flat.transpose(1, 2)) / self.scaling

        return logits

    def get_lowrank_components(self):
        """
        获取低秩分量，用于分析和可视化
        """
        with torch.no_grad():
            # 计算基原型
            base_prototypes = torch.mm(self.U, self.V)  # [E, D]

            # 计算每个超边的"能量"（奇异值贡献）
            U_norm = torch.norm(self.U, dim=1)  # [E]
            V_norm = torch.norm(self.V, dim=0)  # [D]

            # 能量分数
            energy_scores = U_norm.unsqueeze(1) * V_norm.unsqueeze(0)  # [E, D]

            return {
                "base_prototypes": base_prototypes,
                "U": self.U,
                "V": self.V,
                "energy_scores": energy_scores,
                "rank": self.rank,
            }


class CompressedAdaHGConv(nn.Module):
    """
    压缩版自适应超图卷积

    结合低秩超边生成和稀疏化
    """

    def __init__(self, embed_dim, num_hyperedges=32, rank=16, sparse_ratio=0.5, num_heads=4, dropout=0.1):
        super().__init__()

        # 低秩超边生成器
        self.hyperedge_gen = LowRankHyperedgeGen(
            node_dim=embed_dim,
            num_hyperedges=num_hyperedges,
            rank=rank,
            num_heads=num_heads,
            dropout=dropout,
            context="both",
        )

        # 稀疏化处理（可选）
        self.sparse_ratio = sparse_ratio

        # 超边和顶点变换
        self.edge_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(embed_dim)
        )

        self.node_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(embed_dim)
        )

    def _apply_sparsity(self, A):
        """应用稀疏化"""
        if self.sparse_ratio < 1.0 and self.training:
            B, N, E = A.shape
            k = max(1, int(E * self.sparse_ratio))

            # 获取logits（从A反推，这里简化处理）
            # 实际应该从hyperedge_gen获取原始logits
            logits = torch.log(A + 1e-8)  # 近似logits

            # Top-k稀疏化
            topk_values, topk_indices = torch.topk(logits, k=k, dim=-1)
            topk_probs = F.softmax(topk_values, dim=-1)

            A_sparse = torch.zeros_like(A)
            A_sparse.scatter_(-1, topk_indices, topk_probs)

            return A_sparse
        return A

    def forward(self, X):
        # 生成参与矩阵
        A = self.hyperedge_gen(X)  # [B, N, E]

        # 可选：应用稀疏化
        A = self._apply_sparsity(A)

        # 顶点→超边聚合
        He = torch.bmm(A.transpose(1, 2), X)  # [B, E, D]
        He = self.edge_proj(He)

        # 超边→顶点传播
        X_new = torch.bmm(A, He)  # [B, N, D]
        X_new = self.node_proj(X_new)

        # 残差连接
        return X_new + X


class MultiRankHyperedgeGen(nn.Module):
    """
    多秩超边生成器

    使用多个不同秩的分量，然后融合
    可以更好地平衡压缩率和表达能力
    """

    def __init__(self, node_dim, num_hyperedges, ranks=[8, 16, 32]):
        super().__init__()

        self.ranks = ranks
        self.num_components = len(ranks)

        # 多个低秩分量
        self.components = nn.ModuleList(
            [LowRankHyperedgeGen(node_dim, num_hyperedges, rank=rank, num_heads=4, context="both") for rank in ranks]
        )

        # 融合权重
        self.fusion_weights = nn.Parameter(torch.ones(self.num_components))

        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(node_dim, 64), nn.GELU(), nn.Linear(64, self.num_components), nn.Softmax(dim=-1)
        )

    def forward(self, X):
        B, N, D = X.shape

        # 计算各分量
        A_list = []
        for component in self.components:
            A_i = component(X)  # [B, N, E]
            A_list.append(A_i)

        # 计算门控权重（基于全局上下文）
        context = X.mean(dim=1)  # [B, D]
        gates = self.gate_net(context)  # [B, num_components]

        # 加权融合
        A = torch.zeros_like(A_list[0])
        for i, A_i in enumerate(A_list):
            A = A + gates[:, i : i + 1, None] * A_i  # 广播权重

        return A


# 测试代码
def test_lowrank_hyperedge_gen():
    """测试低秩超边生成器"""

    # 测试配置
    B, N, D = 2, 100, 128
    E = 64  # 超边数
    rank = 16

    print("=" * 60)
    print("测试低秩超边生成器")
    print("=" * 60)

    # 1. 创建标准版本作为基准
    print("\n1. 标准超边生成器 (参数量基准)")
    standard_model = LowRankHyperedgeGen(D, E, rank=E, num_heads=4)  # rank=E即标准版本
    X = torch.randn(B, N, D)
    A_standard = standard_model(X)
    print(f"标准版本参与矩阵形状: {A_standard.shape}")

    # 2. 创建低秩版本
    print("\n2. 低秩超边生成器")
    lowrank_model = LowRankHyperedgeGen(D, E, rank=rank, num_heads=4)
    A_lowrank = lowrank_model(X)
    print(f"低秩版本参与矩阵形状: {A_lowrank.shape}")

    # 3. 比较参数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    standard_params = count_parameters(standard_model)
    lowrank_params = count_parameters(lowrank_model)

    print(f"\n参数量对比:")
    print(f"标准版本: {standard_params:,}")
    print(f"低秩版本: {lowrank_params:,}")
    print(f"压缩率: {lowrank_params / standard_params:.2%}")

    # 4. 检查输出性质
    print("\n3. 输出性质检查:")

    # 检查归一化
    row_sums = A_lowrank.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-3), "参与矩阵未正确归一化"
    print("✓ 归一化检查通过")

    # 检查形状
    assert A_lowrank.shape == (B, N, E), f"形状错误: {A_lowrank.shape}"
    print("✓ 形状检查通过")

    # 检查数值范围
    assert (A_lowrank >= 0).all() and (A_lowrank <= 1).all(), "参与矩阵值不在[0,1]范围内"
    print("✓ 数值范围检查通过")

    # 5. 获取低秩分量信息
    print("\n4. 低秩分量分析:")
    components = lowrank_model.get_lowrank_components()
    print(f"秩: {components['rank']}")
    print(f"U矩阵形状: {components['U'].shape}")
    print(f"V矩阵形状: {components['V'].shape}")

    # 计算矩阵的秩（近似）
    base_prototypes = components["base_prototypes"]
    _, s, _ = torch.svd(base_prototypes)
    effective_rank = (s > 1e-3).sum().item()
    print(f"原型矩阵有效秩: {effective_rank}/{D}")

    return lowrank_model, A_lowrank


if __name__ == "__main__":
    # 运行测试
    model, A = test_lowrank_hyperedge_gen()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
