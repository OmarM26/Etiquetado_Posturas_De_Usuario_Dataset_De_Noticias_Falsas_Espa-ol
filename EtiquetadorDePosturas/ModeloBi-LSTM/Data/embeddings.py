# -*- coding: utf-8 -*-
import numpy as np
import torch

def load_embeddings_txt(path: str, stoi: dict, dim: int = None, normalized: bool = False):
    """Carga embeddings en formato .txt/.vec. Devuelve tensor |vocab| x dim."""
    print(f"[INFO] Cargando embeddings desde: {path}")
    if dim is None:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            first = f.readline().strip().split()
            if len(first) <= 5:
                first = f.readline().strip().split()
            dim = len(first) - 1
        print(f"[INFO] Dimensión inferida: {dim}")
    emb = np.random.uniform(-0.05, 0.05, size=(len(stoi), dim)).astype(np.float32)
    covered = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        pos = f.tell()
        header = f.readline().strip().split()
        if not (len(header) == 2 and all(p.isdigit() for p in header)):
            f.seek(pos)
        for line in f:
            parts = line.rstrip().split(' ')
            if len(parts) < dim + 1:
                continue
            word = parts[0]
            vec = np.asarray(parts[1:1+dim], dtype=np.float32)
            if word in stoi:
                emb[stoi[word]] = vec
                covered += 1
    if normalized:
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        emb = emb / norms
    print(f"[INFO] Cobertura de vocab: {covered}/{len(stoi)} ({covered/len(stoi)*100:.2f}%)")
    return torch.tensor(emb)
