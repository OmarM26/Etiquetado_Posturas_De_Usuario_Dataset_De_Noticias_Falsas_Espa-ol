# Scripts/train_pair.py  (versión con métricas y gráfico, sin collate_fn)
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

from Config.config import CFG
from Data.dataset_pair import PairDataset
from Models.bilstm_pair import BiLSTMPair
from transformers import get_linear_schedule_with_warmup


# ----------------------
# Utilidades
# ----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_optimizer(model, lr_head, lr_bert):
    """
    Separa parámetros del encoder (model.enc) y del head.
    - lr_bert  para encoder
    - lr_head  para la cabeza
    """
    encoder_params = list(model.enc.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    head_params = [p for p in model.parameters() if id(p) not in encoder_ids]

    params_with_grad = sum(p.numel() for p in encoder_params if p.requires_grad)
    print(f"Optimizador: {params_with_grad:,} params entrenables en Encoder (lr={lr_bert}), "
          f"{len(head_params)} grupos para Head (lr={lr_head})")

    optimizer_params = [
        {"params": [p for p in encoder_params if p.requires_grad], "lr": lr_bert},
        {"params": head_params, "lr": lr_head},
    ]
    return torch.optim.AdamW(optimizer_params, weight_decay=CFG.WEIGHT_DECAY)


def move_batch_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


# ----------------------
# Train / Eval
# ----------------------
def train_epoch(model, loader, optimizer, criterion, device, scheduler):
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad(set_to_none=True)
        batch = move_batch_to_device(batch, device)

        logits = model(batch)
        loss = criterion(logits, batch["label"])
        loss.backward()

        if getattr(CFG, "MAX_GRAD_NORM", None) is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.MAX_GRAD_NORM)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = batch["label"].size(0)
        total_loss += loss.item() * bs
        total_items += bs

    return total_loss / max(1, total_items)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Evaluating"):
        batch = move_batch_to_device(batch, device)
        logits = model(batch)
        loss = criterion(logits, batch["label"])

        bs = batch["label"].size(0)
        total_loss += loss.item() * bs
        total_items += bs

        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(batch["label"].cpu())

    if len(all_preds) == 0:
        return 0.0, 0.0, 0.0

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    val_loss = total_loss / max(1, total_items)
    val_acc = accuracy_score(labels, preds)
    val_f1m = f1_score(labels, preds, average="macro", zero_division=0)
    return val_loss, val_acc, val_f1m


# ----------------------
# Main
# ----------------------
def main():
    set_seed(CFG.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets
    train_ds = PairDataset(CFG.TRAIN_CSV)
    val_ds   = PairDataset(CFG.VAL_CSV)

    # Sampler balanceado para entrenamiento
    train_labels = train_ds.df["label_id"].to_numpy()
    class_counts = np.bincount(train_labels, minlength=CFG.NUM_CLASSES)
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Loaders (sin collate_fn)
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE,
        sampler=sampler,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.EVAL_BS,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # Modelo
    model = BiLSTMPair().to(device)

    # Optimizador
    optimizer = create_optimizer(model, lr_head=CFG.LR_HEAD, lr_bert=CFG.LR_BERT)

    # Criterio
    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTH)

    # Scheduler: warmup lineal (transformers)
    num_training_steps = len(train_loader) * CFG.EPOCHS
    num_warmup_steps = int(getattr(CFG, "WARMUP_RATIO", 0.1) * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(0, num_warmup_steps),
        num_training_steps=max(1, num_training_steps),
    )
    print(f"Scheduler: warmup_steps={num_warmup_steps} total_steps={num_training_steps}")

    # Tracking de métricas por época
    metrics = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
    }

    os.makedirs("outputs", exist_ok=True)
    best_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, CFG.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{CFG.EPOCHS} ---")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch}: "
              f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | "
              f"Val Acc={val_acc:.4f} | Val Macro-F1={val_f1:.4f}")

        # log
        metrics["epoch"].append(epoch)
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["val_accuracy"].append(val_acc)
        metrics["val_macro_f1"].append(val_f1)

        # checkpoint por mejor F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join("outputs", "best_model.pt"))
            print(f"[Checkpoint] Nuevo mejor F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{CFG.PATIENCE}")
            if patience_counter >= CFG.PATIENCE:
                print("Early stopping.")
                break

    print(f"\nTraining finished. Best Macro-F1: {best_f1:.4f}")

    # Guardar CSV
    csv_path = os.path.join("outputs", "metrics_per_epoch.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"])
        for i in range(len(metrics["epoch"])):
            writer.writerow([
                metrics["epoch"][i],
                f"{metrics['train_loss'][i]:.6f}",
                f"{metrics['val_loss'][i]:.6f}",
                f"{metrics['val_accuracy'][i]:.6f}",
                f"{metrics['val_macro_f1'][i]:.6f}",
            ])
    print(f"[Metrics] CSV guardado en: {csv_path}")

    # Guardar gráfico
    plot_path = os.path.join("outputs", "training_metrics.png")
    plt.figure(figsize=(8, 5))
    epochs = metrics["epoch"]
    plt.plot(epochs, metrics["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, metrics["val_loss"], label="Val Loss", linewidth=2)
    plt.plot(epochs, metrics["val_accuracy"], label="Val Accuracy", linewidth=2)
    plt.plot(epochs, metrics["val_macro_f1"], label="Val Macro-F1", linewidth=2)
    plt.xlabel("Época")
    plt.ylabel("Valor")
    plt.title("Evolución por época: Loss, Accuracy y Macro-F1")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[Plot] Gráfico guardado en: {plot_path}")


if __name__ == "__main__":
    main()
