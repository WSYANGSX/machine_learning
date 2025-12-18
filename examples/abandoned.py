class MFD(nn.Module):
    """Multimodal feature random shielding module"""

    def __init__(self, p=0.15):
        super().__init__()
        self.p = p
        self.alpha = 1.0 / (1.0 - 0.5 * self.p) if self.p > 0 else 1.0

    def forward(self, X: Sequence[list[torch.Tensor]]):
        if not self.training or self.p == 0:
            return X

        if torch.rand(1).item() < self.p:
            drop_idx = random.randint(0, 1)
            keep_idx = 1 - drop_idx

            for i in range(len(X[drop_idx])):
                X[drop_idx][i] = torch.zeros_like(X[drop_idx][i])

            for i in range(len(X[keep_idx])):
                X[keep_idx][i] = X[keep_idx][i] * self.alpha
        else:
            for modality in X:
                for i in range(len(modality)):
                    modality[i] = modality[i] * self.alpha

        return X


# TODO Whether to add an asymmetric prototype
class CrossGraphHyperedgeGen(nn.Module):
    """
    Output when sparse:
        - b_idx: [B, N A, kB] B neighbor index selected by node A.
        - A_w: [B, N A, E, kB] corresponding participation weight (softmax over E).
        - N_B: int B graph node number (for optional densify visualization) still return A cross.

    Output when dense:
        - A_cross: [B, NA, E, NB].
    """

    def __init__(
        self,
        node_dim: int,
        num_hyperedges: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0,
        sparse_ratio: float = 1.0,  # 稀疏在 N_B 上做 top-kB
        use_learnable_temperature: bool = False,
        return_dense_for_viz: bool = False,  # ✅只有调试可视化才置True
    ):
        super().__init__()
        assert 0 < sparse_ratio <= 1, f"sparse_ratio must be in (0, 1], got {sparse_ratio}"
        assert node_dim % num_heads == 0, f"node_dim {node_dim} must be divisible by num_heads {num_heads}"

        self.node_dim = node_dim
        self.num_hyperedges = num_hyperedges
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.sparse_ratio = sparse_ratio
        self.return_dense_for_viz = return_dense_for_viz

        self.cross_prototypes = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.cross_prototypes)

        self.proj_A = nn.Linear(node_dim, node_dim)
        self.proj_B = nn.Linear(node_dim, node_dim)

        self.cross_attention_net = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, num_heads),
            nn.Softmax(dim=-1),
        )

        self.temperature = nn.Parameter(torch.tensor(temperature)) if use_learnable_temperature else temperature
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    @staticmethod
    def densify(b_idx: torch.Tensor, A_w: torch.Tensor, N_B: int) -> torch.Tensor:
        """仅调试可视化用：把稀疏表示还原成 dense A_cross [B,N_A,E,N_B]"""
        B, N_A, E, kB = A_w.shape
        A_cross = A_w.new_zeros(B, N_A, E, N_B)
        scatter_idx = b_idx.unsqueeze(2).expand(-1, -1, E, -1)  # [B,N_A,E,kB]
        A_cross.scatter_(3, scatter_idx, A_w)
        return A_cross

    def forward(self, X_A: torch.Tensor, X_B: torch.Tensor):
        B, N_A, D = X_A.shape
        _, N_B, _ = X_B.shape
        H = self.num_heads
        E = self.num_hyperedges
        d_h = self.head_dim

        X_A_proj = self.proj_A(X_A)  # [B,N_A,D]
        X_B_proj = self.proj_B(X_B)  # [B,N_B,D]

        Q = X_A_proj.view(B, N_A, H, d_h).transpose(1, 2).contiguous()  # [B,H,N_A,d_h]
        K = X_B_proj.view(B, N_B, H, d_h).transpose(1, 2).contiguous()  # [B,H,N_B,d_h]

        # head weights: [B,H]
        cross_context = torch.cat([X_A.mean(dim=1), X_B.mean(dim=1)], dim=-1)  # [B,2D]
        head_w = self.cross_attention_net(cross_context)  # [B,H]

        # node-prototype sim: [B,H,N_A,E]
        proto = self.cross_prototypes.view(E, H, d_h).permute(1, 0, 2).contiguous()  # [H,E,d_h]
        Q_flat = Q.reshape(B * H, N_A, d_h)  # [BH,N_A,d_h]
        P_flat = proto.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * H, E, d_h)  # [BH,E,d_h]
        node_proto_sim = torch.bmm(Q_flat, P_flat.transpose(1, 2)).view(B, H, N_A, E)  # [B,H,N_A,E]

        # dense分支（sparse_ratio==1 且需要dense语义）
        if self.sparse_ratio >= 1:
            cross_similarity = torch.matmul(Q, K.transpose(-1, -2)) / self.scaling  # [B,H,N_A,N_B]
            logits = node_proto_sim.unsqueeze(-1) * cross_similarity.unsqueeze(3)  # [B,H,N_A,E,N_B]
            logits = logits * head_w.view(B, H, 1, 1, 1)
            logits = logits.sum(dim=1)  # [B,N_A,E,N_B]
            logits = self.dropout(logits / self.temperature)
            A_cross = F.softmax(logits, dim=2)
            return A_cross

        # ------------------- 真稀疏：只在 N_B 上做 top-kB，不再scatter回full -------------------
        kB = max(1, int(N_B * self.sparse_ratio))

        # 1) 先算“用于选邻居”的 weighted cross score: [B,N_A,N_B]
        # 不存 cross_similarity[B,H,N_A,N_B]，按head累加，峰值省一倍H
        cross_score = Q.new_zeros(B, N_A, N_B)
        for h in range(H):
            score_h = torch.matmul(Q[:, h], K[:, h].transpose(1, 2)) / self.scaling  # [B,N_A,N_B]
            cross_score = cross_score + head_w[:, h].view(B, 1, 1) * score_h

        _, b_idx = torch.topk(cross_score, k=kB, dim=-1)  # [B,N_A,kB]

        # 2) 对每个head，仅对选中的kB做点积，不构造全N_B
        topi_rep = b_idx.unsqueeze(1).expand(B, H, N_A, kB).reshape(B * H, N_A, kB)  # [BH,N_A,kB]
        K_flat = K.reshape(B * H, N_B, d_h)  # [BH,N_B,d_h]

        # gather K_sel: [BH,N_A,kB,d_h]（expand是view，不会实际分配[BH,N_A,N_B,d_h]）
        K_flat4 = K_flat.unsqueeze(1).expand(B * H, N_A, N_B, d_h)
        idx_flat = topi_rep.unsqueeze(-1).expand(B * H, N_A, kB, d_h)
        K_sel = K_flat4.gather(2, idx_flat)

        score_sel = (Q_flat.unsqueeze(2) * K_sel).sum(-1) / self.scaling  # [BH,N_A,kB]
        cross_sim_sparse = score_sel.view(B, H, N_A, kB)  # [B,H,N_A,kB]

        # 3) logits_sparse: [B,N_A,E,kB]
        logits_sparse = node_proto_sim.unsqueeze(-1) * cross_sim_sparse.unsqueeze(3)  # [B,H,N_A,E,kB]
        logits_sparse = logits_sparse * head_w.view(B, H, 1, 1, 1)
        logits_sparse = logits_sparse.sum(dim=1)  # [B,N_A,E,kB]
        logits_sparse = self.dropout(logits_sparse / self.temperature)

        A_w = F.softmax(logits_sparse, dim=2)  # softmax over E -> [B,N_A,E,kB]

        if self.return_dense_for_viz:
            return self.densify(b_idx, A_w, N_B)

        return b_idx, A_w, N_B


class CrossGraphHyperConv(nn.Module):
    """
    Cross-graph hypergraph convolution.
    Performs message passing between feature map A and feature map B.
    """

    def __init__(
        self,
        embed_dim: int,
        num_hyperedges: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
        sparse_ratio: float = 0.5,
        bidirectional: bool = True,
        fusion_type: Literal["bilinear", "concatenate", "gate"] = "bilinear",
        return_dense_for_viz: bool = False,  # for test
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_hyperedges = num_hyperedges
        self.bidirectional = bidirectional
        self.fusion_type = fusion_type

        self.cross_gen = CrossGraphHyperedgeGen(
            node_dim=embed_dim,
            num_hyperedges=num_hyperedges,
            num_heads=num_heads,
            dropout=dropout,
            sparse_ratio=sparse_ratio,
            return_dense_for_viz=return_dense_for_viz,
        )

        self.A_to_B_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout))
        self.B_to_A_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout))

        if fusion_type == "gate":
            self.gate_A = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
            self.gate_B = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        elif fusion_type == "bilinear":
            self.bilinear_fusion = nn.Bilinear(embed_dim, embed_dim, embed_dim)

    @staticmethod
    def _safe_norm_over_E(A_node_edge: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        A_node_edge: [B,N,E]，归一化到sum_E=1；若全0行则退化为均匀分布
        """
        den = A_node_edge.sum(dim=2, keepdim=True)  # [B,N,1]
        E = A_node_edge.size(2)
        return torch.where(
            den > 0,
            A_node_edge / (den + eps),
            A_node_edge.new_full(A_node_edge.shape, 1.0 / E),
        )

    @staticmethod
    def _gather_neighbors(X_src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        X_src: [B,N_src,D]
        idx  : [B,N_tgt,k]
        return: [B,N_tgt,k,D]
        """
        B, N_src, D = X_src.shape
        _, N_tgt, k = idx.shape
        X_exp = X_src.unsqueeze(1).expand(B, N_tgt, N_src, D)  # view
        gather_idx = idx.unsqueeze(-1).expand(B, N_tgt, k, D)  # [B,N_tgt,k,D]
        return X_exp.gather(2, gather_idx)  # [B,N_tgt,k,D]

    def forward(self, X_A: torch.Tensor, X_B: torch.Tensor):
        out = self.cross_gen(X_A, X_B)

        # -------- dense路径（仅当return_dense_for_viz=True或sparse_ratio==1）--------
        if isinstance(out, torch.Tensor):
            A_cross = out  # [B,N_A,E,N_B]

            He_B = torch.einsum("bnem,bmd->bned", A_cross, X_B)  # [B,N_A,E,D]
            He_B_t = self.A_to_B_proj(He_B)
            A_node_edge = self._safe_norm_over_E(A_cross.sum(dim=3))  # [B,N_A,E]
            X_A_from_B = torch.einsum("bne,bned->bnd", A_node_edge, He_B_t)

            if self.bidirectional:
                A_cross_T = A_cross.transpose(1, 3)  # [B,N_B,E,N_A]
                He_A = torch.einsum("bnem,bmd->bned", A_cross_T, X_A)  # [B,N_B,E,D]
                He_A_t = self.B_to_A_proj(He_A)
                A_node_edge_T = self._safe_norm_over_E(A_cross_T.sum(dim=3))
                X_B_from_A = torch.einsum("bne,bned->bnd", A_node_edge_T, He_A_t)
            else:
                X_B_from_A = X_B

            cross_repr = A_cross
        else:
            # -------- sparse路径：不构造 full A_cross --------
            b_idx, A_w, N_B = out  # b_idx:[B,N_A,kB], A_w:[B,N_A,E,kB]

            X_B_sel = self._gather_neighbors(X_B, b_idx)  # [B,N_A,kB,D]
            He_B = torch.einsum("bnek,bnkd->bned", A_w, X_B_sel)  # [B,N_A,E,D]
            He_B_t = self.A_to_B_proj(He_B)

            A_node_edge = self._safe_norm_over_E(A_w.sum(dim=3))  # [B,N_A,E]
            X_A_from_B = torch.einsum("bne,bned->bnd", A_node_edge, He_B_t)

            if self.bidirectional:
                # ✅再生成一次 B->A 的稀疏关系（替代 transpose dense）
                a_out = self.cross_gen(X_B, X_A)
                if isinstance(a_out, torch.Tensor):
                    # 理论上只有你把return_dense_for_viz开了才会进来
                    A_cross_BA = a_out
                    He_A = torch.einsum("bnem,bmd->bned", A_cross_BA, X_A)
                    He_A_t = self.B_to_A_proj(He_A)
                    A_node_edge_T = self._safe_norm_over_E(A_cross_BA.sum(dim=3))
                    X_B_from_A = torch.einsum("bne,bned->bnd", A_node_edge_T, He_A_t)
                else:
                    a_idx, A_w_BA, N_A = a_out  # a_idx:[B,N_B,kA], A_w_BA:[B,N_B,E,kA]
                    X_A_sel = self._gather_neighbors(X_A, a_idx)  # [B,N_B,kA,D]
                    He_A = torch.einsum("bnek,bnkd->bned", A_w_BA, X_A_sel)  # [B,N_B,E,D]
                    He_A_t = self.B_to_A_proj(He_A)
                    A_node_edge_T = self._safe_norm_over_E(A_w_BA.sum(dim=3))
                    X_B_from_A = torch.einsum("bne,bned->bnd", A_node_edge_T, He_A_t)
            else:
                X_B_from_A = X_B

            cross_repr = (b_idx, A_w, N_B)  # ✅稀疏表示，必要时可用 CrossGraphHyperedgeGen.densify 还原

        # -------- fusion --------
        if self.fusion_type == "concatenate":
            X_A_new = torch.cat([X_A, X_A_from_B], dim=-1)
            X_B_new = torch.cat([X_B, X_B_from_A], dim=-1)
        elif self.fusion_type == "gate":
            gate_A = self.gate_A(torch.cat([X_A, X_A_from_B], dim=-1))
            X_A_new = gate_A * X_A + (1 - gate_A) * X_A_from_B
            gate_B = self.gate_B(torch.cat([X_B, X_B_from_A], dim=-1))
            X_B_new = gate_B * X_B + (1 - gate_B) * X_B_from_A
        elif self.fusion_type == "bilinear":
            X_A_new = self.bilinear_fusion(X_A, X_A_from_B)
            X_B_new = self.bilinear_fusion(X_B, X_B_from_A)
        else:
            X_A_new = X_A + X_A_from_B
            X_B_new = X_B + X_B_from_A

        return X_A_new, X_B_new, cross_repr


class CrossGraphHyperConv(nn.Module):
    """
    Cross-graph hypergraph convolution.
    Performs message passing between feature map A and feature map B.
    """

    def __init__(
        self,
        embed_dim: int,
        num_hyperedges: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
        sparse_ratio=0.5,
        bidirectional: bool = True,
        fusion_type: Literal["bilinear", "concatenate", "gate"] = "bilinear",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_hyperedges = num_hyperedges
        self.bidirectional = bidirectional
        self.fusion_type = fusion_type

        # Cross-graph hyperedge generator
        self.cross_gen = CrossGraphHyperedgeGen(
            node_dim=embed_dim,
            num_hyperedges=num_hyperedges,
            num_heads=num_heads,
            dropout=dropout,
            sparse_ratio=sparse_ratio,
        )

        # Message passing function
        self.A_to_B_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.B_to_A_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Integrated access control
        if fusion_type == "gate":
            self.gate_A = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
            self.gate_B = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        elif fusion_type == "bilinear":
            self.bilinear_fusion = nn.Bilinear(embed_dim, embed_dim, embed_dim)

    def forward(self, X_A: torch.Tensor, X_B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform bidirectional cross-graph message passing

        Args:
            X_A: [B, N_A, D] map A.
            X_B: [B, N_B, D] map B.

        Returns:
            X_A_new: [B, N_A, D] The updated feature map A.
            X_B_new: [B, N_B, D] The updated feature map B.
        """
        # A_cross: [B, N_A, E, N_B]
        A_cross = self.cross_gen(X_A, X_B)

        # Aggregate the features of B to the hyperedge for each node in A
        He_B = torch.einsum("bnem,bmd->bned", A_cross, X_B)  # [B, N_A, E, D]
        He_B_transformed = self.A_to_B_proj(He_B)

        A_node_edge = A_cross.sum(dim=3)  # [B, N_A, E]
        A_node_edge = A_node_edge / (A_node_edge.sum(dim=2, keepdim=True) + 1e-6)  # sparse and then normalized

        # Propagate from the hyperedge to the node of A
        X_A_from_B = torch.einsum("bne,bned->bnd", A_node_edge, He_B_transformed)  # [B, N_A, D]

        # Bidirectional transmission: Message transmission from A to B
        if self.bidirectional:
            # Use the transposed participation matrix
            A_cross_T = A_cross.transpose(1, 3)  # [B, N_B, E, N_A]

            # Aggregate the features of A to the hyperedge for each node in B
            He_A = torch.einsum("bnem,bmd->bned", A_cross_T, X_A)  # [B, N_B, E, D]
            He_A_transformed = self.B_to_A_proj(He_A)

            A_node_edge_T = A_cross_T.sum(dim=3)  # [B, N_B, E]
            A_node_edge_T = A_node_edge_T / (A_node_edge_T.sum(dim=2, keepdim=True) + 1e-6)

            # The node propagated from the hyperedge to B
            X_B_from_A = torch.einsum("bne,bned->bnd", A_node_edge_T, He_A_transformed)  # [B, N_B, D]
        else:
            X_B_from_A = X_B  # If it is not two-way, keep it as it is

        # Feature fusion
        if self.fusion_type == "concatenate":
            X_A_new = torch.cat([X_A, X_A_from_B], dim=-1)
            X_B_new = torch.cat([X_B, X_B_from_A], dim=-1)
        elif self.fusion_type == "gate":
            # Door control integration
            gate_A = self.gate_A(torch.cat([X_A, X_A_from_B], dim=-1))
            X_A_new = gate_A * X_A + (1 - gate_A) * X_A_from_B
            gate_B = self.gate_B(torch.cat([X_B, X_B_from_A], dim=-1))
            X_B_new = gate_B * X_B + (1 - gate_B) * X_B_from_A
        elif self.fusion_type == "bilinear":
            X_A_new = self.bilinear_fusion(X_A, X_A_from_B)
            X_B_new = self.bilinear_fusion(X_B, X_B_from_A)
        else:
            # Residual connection
            X_A_new = X_A + X_A_from_B
            X_B_new = X_B + X_B_from_A

        return X_A_new, X_B_new, A_cross  # Return the participation matrix for visualization
