import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from Config.config import CFG
from Data.tokenizer import get_tokenizer

def _label_to_idx(raw):
    lab = str(raw).strip().lower()
    mapping = {name.lower(): i for i, name in enumerate(CFG.LABELS)}
    if lab in mapping:
        return mapping[lab]
    try:
        y = int(lab)
        if 1 <= y <= CFG.NUM_CLASSES: return y - 1
        return y
    except Exception as e:
        raise ValueError(f"Etiqueta desconocida: {raw}") from e

class PairDataset(Dataset):
    def __init__(self, csv_path, cache_split=None):
        self.df = pd.read_csv(csv_path)

        self.label2id = {
            "comment": 0,
            "support": 1,
            "query":   2,
            "deny":    3,
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

        lbl_col = getattr(CFG, "LABEL_COL", "label_sdqc")
        if lbl_col not in self.df.columns:
            raise ValueError(f"Columna de etiqueta '{lbl_col}' no existe en {csv_path}")

        self.df[lbl_col] = self.df[lbl_col].astype(str).str.strip().str.lower()

        unknown = sorted(set(self.df[lbl_col].unique()) - set(self.label2id.keys()))
        if unknown:
            raise ValueError(f"Etiquetas desconocidas en '{lbl_col}': {unknown}. "
                             f"Esperadas: {list(self.label2id.keys())}")

        self.df["label_id"] = self.df[lbl_col].map(self.label2id).astype(int)

        self.tokenizer = get_tokenizer()
        self.cache_split = cache_split
        self.use_cache = CFG.CACHE_EMBEDS and (self.cache_split is not None)

        self.monolith = None
        self.monolith_ok = False
        if self.use_cache and getattr(CFG, "CACHE_FORMAT", "sharded") == "monolith":
            monopath = os.path.join(CFG.CACHE_DIR, f"{self.cache_split}_embeds.pt")
            if os.path.exists(monopath):
                self.monolith = torch.load(monopath, map_location="cpu")
                for k in ["rep", "par", "rep_mask", "par_mask"]:
                    if k not in self.monolith:
                        raise RuntimeError(f"Falta clave '{k}' en {monopath}")
                self.monolith_ok = True

    def __len__(self): return len(self.df)

    def _tokenize(self, texts):
        return self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=CFG.MAX_LEN, return_tensors="pt"
        )

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        # Usamos la columna 'label_id' que creamos en __init__ para consistencia
        label = int(row["label_id"])

        if self.use_cache and self.monolith_ok:
            rep = self.monolith["rep"][idx]
            par = self.monolith["par"][idx]
            rmk = self.monolith["rep_mask"][idx]
            pmk = self.monolith["par_mask"][idx]
            return {
                "rep_emb": rep, "par_emb": par,
                "rep_mask": rmk.bool(), "par_mask": pmk.bool(),
                "label": label
            }

        # Fallback: tokeniza en vivo
        reply  = str(row[CFG.TEXT_COL_REPLY])
        parent = str(row[CFG.TEXT_COL_PARENT])
        toks_r = self._tokenize([reply])
        toks_p = self._tokenize([parent])
        return {
            "rep_input_ids": toks_r["input_ids"].squeeze(0),
            "rep_attention_mask": toks_r["attention_mask"].squeeze(0),
            "par_input_ids": toks_p["input_ids"].squeeze(0),
            "par_attention_mask": toks_p["attention_mask"].squeeze(0),
            "label": label,
            "idx": idx
        }