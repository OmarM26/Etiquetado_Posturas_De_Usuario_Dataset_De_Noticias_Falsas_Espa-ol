# Scripts/build_pairs_from_spanish.py
import sys, os, pandas as pd

SPANISH_TO_EN = {"comenta":"comment","valida":"support","consulta":"query","niega":"deny"}
NEEDED = ["texto_respuesta","texto_padre","texto_raiz","postura"]

def main():
    if len(sys.argv) < 3:
        print("Uso: python -m Scripts.build_pairs_from_spanish <input_es_csv> <output_pairs_csv>")
        raise SystemExit(1)
    src, dst = sys.argv[1], sys.argv[2]
    if not os.path.isfile(src):
        print("No encuentro:", src); raise SystemExit(1)

    df = pd.read_csv(src, encoding="utf-8", sep=None, engine="python")
    miss = [c for c in NEEDED if c not in df.columns]
    if miss:
        print("Faltan columnas:", miss); raise SystemExit(1)

    lbl = df["postura"].astype(str).str.strip().str.lower().map(SPANISH_TO_EN)
    ok = lbl.notna()
    if ok.mean() < 0.95:
        print("Demasiadas etiquetas no mapeadas. Valores únicos en 'postura':")
        print(df["postura"].astype(str).str.strip().value_counts())
        raise SystemExit(1)

    out = pd.DataFrame({
        "reply":  df["texto_respuesta"].astype(str),
        "parent": df["texto_padre"].astype(str),
        "root":   df["texto_raiz"].astype(str),
        "label_sdqc": lbl
    })
    out = out.loc[out["label_sdqc"].notna()].copy()

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    out.to_csv(dst, index=False, encoding="utf-8")
    print(f"[OK] Guardado: {dst}")
    print(out["label_sdqc"].value_counts())

if __name__ == "__main__":
    main()
