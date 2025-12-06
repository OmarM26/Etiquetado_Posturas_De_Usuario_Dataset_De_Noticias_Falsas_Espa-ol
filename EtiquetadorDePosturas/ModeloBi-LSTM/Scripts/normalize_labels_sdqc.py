# Scripts/normalize_labels_sdqc.py
import sys, os
import pandas as pd

SPANISH_TO_EN = {
    "comenta": "comment",
    "valida": "support",
    "consulta": "query",
    "niega": "deny",
}

# Dos posibles esquemas numéricos:
NUM_TO_EN_COMENTA0 = {0: "comment", 1: "deny", 2: "support", 3: "query"}   # tu caso más probable (comenta mayoritaria)
NUM_TO_EN_SDQC0123 = {0: "support", 1: "deny", 2: "query", 3: "comment"}   # SDQC clásico

VALID = {"comment", "support", "query", "deny"}

def infer_numeric_scheme(series):
    """Si la clase 0 domina, asumimos COMENTA0; de lo contrario SDQC clásico."""
    vc = series.value_counts(dropna=True)
    if vc.empty:
        return NUM_TO_EN_COMENTA0
    dominant = vc.idxmax()
    # Si el valor dominante es 0, suele corresponder a 'comenta' en tu archivo
    if dominant == 0:
        return NUM_TO_EN_COMENTA0
    return NUM_TO_EN_SDQC0123

def main():
    if len(sys.argv) < 3:
        print("Uso: python -m Scripts.normalize_labels_sdqc <input_csv> <output_csv>")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    if not os.path.isfile(src):
        print(f"No encuentro: {src}"); sys.exit(1)

    # Leer con tu separador ya corregido (;) o auto
    try:
        df = pd.read_csv(src, encoding="utf-8", sep=None, engine="python")
    except Exception:
        df = pd.read_csv(src, encoding="utf-8-sig", sep=";", engine="python")

    if "label_sdqc" not in df.columns:
        print(f"Falta 'label_sdqc' en columnas: {df.columns.tolist()}")
        sys.exit(1)

    # Detectar tipo de label: str (es/inglés) o numérico
    col = df["label_sdqc"]

    # 1) Intento: mapear desde español/inglés en string
    if col.dtype == object:
        mapped = (
            col.astype(str).str.strip().str.lower()
              .map(SPANISH_TO_EN)  # español -> inglés
              .fillna(col.astype(str).str.strip().str.lower())  # si ya estaba en inglés
        )
        # Validar
        ok_mask = mapped.isin(VALID)
        # Si muy pocos son válidos, intentamos tratar como números escritos como string
        if ok_mask.mean() < 0.5:
            # intentar convertir a int
            try:
                nums = pd.to_numeric(col, errors="coerce")
                scheme = infer_numeric_scheme(nums.dropna().astype(int))
                mapped = nums.map(scheme)
                ok_mask = mapped.isin(VALID)
            except Exception:
                pass
    else:
        # 2) Numérico -> inglés
        nums = pd.to_numeric(col, errors="coerce")
        scheme = infer_numeric_scheme(nums.dropna().astype(int))
        mapped = nums.map(scheme)
        ok_mask = mapped.isin(VALID)

    # Drop NaN / no mapeados
    dropped = (~ok_mask).sum()
    if dropped > 0:
        print(f"[INFO] Filas sin etiqueta válida (se eliminan): {dropped}")
    df = df.loc[ok_mask].copy()
    df["label_sdqc"] = mapped.loc[ok_mask]

    # Verificar columnas de texto requeridas
    needed = ["reply", "parent", "root", "label_sdqc"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"Faltan columnas {missing}. Presentes: {df.columns.tolist()}")
        sys.exit(1)

    # Validar dominio final
    bad = ~df["label_sdqc"].isin(VALID)
    if bad.any():
        print(f"Quedaron etiquetas inválidas: {df.loc[bad, 'label_sdqc'].unique().tolist()}")
        sys.exit(1)

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    df.to_csv(dst, index=False, encoding="utf-8")
    print(f"[OK] Normalizado y guardado en: {dst}")
    print("Distribución final:\n", df["label_sdqc"].value_counts())

if __name__ == "__main__":
    main()
