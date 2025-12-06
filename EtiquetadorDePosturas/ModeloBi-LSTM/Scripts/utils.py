import os, random, numpy as np, torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from Config.config import CFG

def seed_everything(seed=CFG.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = CFG.CUDNN_BENCHMARK

def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def compute_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def class_weights_from_counts(counts):
    import numpy as np, torch
    counts = np.array(counts, dtype=np.float32)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
