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
