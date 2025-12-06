import torch, torch.nn as nn, torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        if self.label_smoothing and self.label_smoothing > 0:
            num_classes = logits.size(-1)
            with torch.no_grad():
                true_dist = torch.zeros_like(logits)
                true_dist.fill_(self.label_smoothing / (num_classes - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1 - self.label_smoothing)
            log_probs = torch.log_softmax(logits, dim=-1)
            ce = -(true_dist * log_probs).sum(dim=-1)
        else:
            ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")

        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
