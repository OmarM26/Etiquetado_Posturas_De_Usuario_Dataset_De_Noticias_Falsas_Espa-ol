import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# CONFIGURACIÓN
# ============================================================

# Ruta a la carpeta del modelo
RUTA_MODELO = "Z:/modelos_nuevos/modelo_final_contexto_postmod256_5e5_0.7317"

# Ruta a tu dataset de hilos
RUTA_BASE = Path("C:/Users/omarm/Downloads/estado del arte papers/datasets/DataEtiquetada")


# ============================================================
# CARGA DEL MODELO
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cargando modelo desde:", RUTA_MODELO)

tokenizer = AutoTokenizer.from_pretrained(RUTA_MODELO)
model = AutoModelForSequenceClassification.from_pretrained(RUTA_MODELO)
model = model.to(device)
model.eval()

id2label = model.config.id2label
print("Modelo cargado. Etiquetas:", id2label)


# ============================================================
# FUNCIÓN DE PREDICCIÓN
# ============================================================

@torch.no_grad()
def predecir_stance(texto: str) -> str:
    """
    Devuelve la etiqueta de postura directamente en español
    (Comenta, Consulta, De acuerdo, Desacuerdo).
    """
    inputs = tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=False
    ).to(device)

    logits = model(**inputs).logits
    pred_idx = torch.argmax(logits, dim=1).item()

    # 🔴 AQUÍ ESTABA EL PROBLEMA: antes usábamos str(pred_idx)
    etiqueta = id2label[pred_idx]

    return etiqueta


# ============================================================
# RECORRIDO DE TWEETS (RECURSIVO)
# ============================================================

def etiquetar_tweet(tweet: dict):
    """
    Etiqueta recursivamente un tweet y todos sus hijos en 'Tweets',
    sobrescribiendo el campo 'postura'.
    """
    texto = tweet.get("texto")
    if texto:
        tweet["postura"] = predecir_stance(texto)

    hijos = tweet.get("Tweets", [])
    if isinstance(hijos, list):
        for hijo in hijos:
            if isinstance(hijo, dict):
                etiquetar_tweet(hijo)
    # Retweets se dejan igual


def etiquetar_archivo_hilo(ruta_json: Path):
    """
    Abre el JSON de un hilo, etiqueta todos los tweets dentro de "Tweets"
    y sobrescribe el mismo archivo.
    """
    print(f"Etiquetando hilo: {ruta_json}")

    with ruta_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    lista_tweets = data.get("Tweets", [])
    if isinstance(lista_tweets, list):
        for tweet in lista_tweets:
            if isinstance(tweet, dict):
                etiquetar_tweet(tweet)

    with ruta_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def recorrer_y_etiquetar_todos(ruta_base: Path):
    """
    Recorre todas las subcarpetas (cada una es un hilo) y etiqueta su .json.
    """
    if not ruta_base.exists():
        raise FileNotFoundError(f"No existe la ruta base: {ruta_base}")

    # Detectar tweet_raiz.json si existe (solo informativo)
    for nombre in ["tweet_raiz.json", "tweet_raíz.json"]:
        ruta_ref = ruta_base / nombre
        if ruta_ref.exists():
            print(f"Archivo de referencias encontrado: {ruta_ref}")
            break

    for carpeta in ruta_base.iterdir():
        if not carpeta.is_dir():
            continue

        id_hilo = carpeta.name
        ruta_json = carpeta / f"{id_hilo}.json"

        if ruta_json.exists():
            etiquetar_archivo_hilo(ruta_json)
        else:
            print(f"⚠ No se encontró {id_hilo}.json dentro de {carpeta}")


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    print(f"Procesando dataset en: {RUTA_BASE}")
    recorrer_y_etiquetar_todos(RUTA_BASE)
    print("✅ Etiquetado completo.")
