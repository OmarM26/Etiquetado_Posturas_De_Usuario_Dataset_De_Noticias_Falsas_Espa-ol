import torch, torch.nn as nn

class AttnPool(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.w = nn.Linear(hidden, 1)

    def forward(self, H, mask):
        # H: [B,T,H], mask: [B,T] (bool)
        scores = self.w(H).squeeze(-1)  # [B,T]
        # Evitar overflow en FP16: usar un valor finito muy negativo dentro del rango de float16
        neg_inf = -1e4 if scores.dtype == torch.float16 else -1e9
        scores = scores.masked_fill(~mask, neg_inf)
        alpha = torch.softmax(scores, dim=-1)
        z = (H * alpha.unsqueeze(-1)).sum(dim=1)  # [B,H]
        return z, alpha
