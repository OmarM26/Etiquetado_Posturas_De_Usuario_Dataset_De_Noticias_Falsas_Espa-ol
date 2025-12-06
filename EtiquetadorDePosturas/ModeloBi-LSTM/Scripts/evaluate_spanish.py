# Eval/evaluate_pairs_cli.py
import os
import json
import argparse
from typing import List

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt


from Data.dataset_pair import PairDataset
from Models.bilstm_pair import BiLSTMPair

from Config.config import CFG
# AÑADE ESTO justo después:
CFG.TEXT_COL_REPLY  = "reply"
CFG.TEXT_COL_PARENT = "parent"
CFG.TEXT_COL_ROOT   = "root"
CFG.LABEL_COL       = "label_sdqc"

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluación de modelo Pair (transfer a español) con ruta de checkpoint explícita."
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ruta al checkpoint .pt/.pth a evaluar (state_dict).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="CSV de evaluación con columnas: reply, parent, root, label_id.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño de batch para evaluación (por defecto 32).",
    )
    p.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Dispositivo: auto/cpu/cuda (default: auto).",
    )
    p.add_argument(
        "--target-names",
        type=str,
        default="",
        help=(
            "Lista separada por coma con los nombres de clases en orden de label_id. "
            "Ej: 'Comenta,Niega,Valida,Consulta' o 'Valida,Niega,Consulta,Comenta'. "
            "Si no se pasa, no imprime nombres amigables."
        ),
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default="./outputs_eval",
        help="Carpeta donde guardar reporte JSON y matriz de confusión (PNG).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Usar strict=True al cargar state_dict (por defecto False, tolerante a claves faltantes/extra).",
    )
    return p.parse_args()


def get_device(name: str):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def main():
    args = parse_args()
    device = get_device(args.device)

    # 1) Dataset y DataLoader (usa tu PairDataset tal cual)
    if not os.path.isfile(args.dataset):
        raise FileNotFoundError(f"No encuentro el dataset: {args.dataset}")
    test_ds = PairDataset(args.dataset)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=getattr(CFG, "NUM_WORKERS", 0),
        pin_memory=(device.type == "cuda"),
    )

    # 2) Modelo
    model = BiLSTMPair().to(device)

    # 3) Cargar checkpoint de forma robusta
    ckpt_path = args.model
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No encuentro el checkpoint: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned = {}
    if isinstance(state, dict):
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            cleaned[nk] = v
    else:
        raise ValueError("El checkpoint no es un dict de state_dict conocido.")

    missing, unexpected = model.load_state_dict(cleaned, strict=args.strict)
    if missing:
        print("⚠️ Missing keys:", missing)
    if unexpected:
        print("⚠️ Unexpected keys:", unexpected)
    model.eval()

    # 4) Inference
    y_true, y_pred = [], []
    for batch in test_loader:
        # mover tensores al device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        logits = model(batch)
        y_true.extend(batch["label"].cpu().numpy().tolist())
        y_pred.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 5) Métricas
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # target names (opcional)
    target_names: List[str] = []
    if args.target_names.strip():
        target_names = [t.strip() for t in args.target_names.split(",")]

    print("\n=== RESULTADOS EVALUACIÓN ===")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Dataset   : {args.dataset}")
    print(f"Device    : {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Macro-F1  : {f1m:.4f}")
    if target_names:
        print(
            "\nClassification report:\n",
            classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                digits=4,
                zero_division=0,
            ),
        )
    else:
        print(
            "\nClassification report (sin target_names):\n",
            classification_report(y_true, y_pred, digits=4, zero_division=0),
        )

    # 6) Guardar artefactos: JSON + matriz de confusión PNG
    os.makedirs(args.save_dir, exist_ok=True)

    results = {
        "model_path": ckpt_path,
        "dataset_path": args.dataset,
        "batch_size": args.batch_size,
        "device": str(device),
        "accuracy": float(acc),
        "macro_f1": float(f1m),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "num_samples": int(len(y_true)),
        "class_names": target_names if target_names else None,
    }
    with open(os.path.join(args.save_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[OK] Guardado JSON: {os.path.join(args.save_dir, 'eval_results.json')}")

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    fig_path = os.path.join(args.save_dir, "confusion_matrix.png")
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de confusión")
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    labels = target_names if target_names else [str(i) for i in range(cm.shape[0])]
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    # anotar valores
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[OK] Guardado CM: {fig_path}")


if __name__ == "__main__":
    main()
