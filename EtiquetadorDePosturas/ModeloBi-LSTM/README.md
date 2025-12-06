# BiLSTM Pair + (Distil)BERT Patch — 2025-09-16

Este parche incluye:
- Congelamiento opcional de BERT y **cacheo de embeddings** para entrenamiento turbo.
- **BiLSTM Pair** con **attention pooling** y features de pares `[zr, zp, |zr−zp|, zr*zp]`.
- **FocalLoss** y **WeightedRandomSampler** para desbalance severo.
- **AMP (FP16)**, OneCycleLR, grad clipping, early stopping por **macro-F1**.
- Control total vía `Config/CFG` (sin flags por consola).

## Estructura
```
Config/
  config.py
Data/
  dataset_pair.py
  tokenizer.py
Eval/
  evaluate_pair.py
Models/
  attn_pool.py
  bilstm_pair.py
  losses.py
Predict/
  infer_pair.py
Scripts/
  train_pair.py
  precompute_embeds.py
  utils.py
```
> Nota: Puedes integrar estos archivos en tu repo; respeta los mismos nombres de carpeta.

## Requisitos
- Python 3.9+
- PyTorch (con CUDA para tu GTX 1660 SUPER)
- `transformers`, `scikit-learn`, `pandas`, `numpy`

## Dataset esperado (mínimo)
Un CSV (train/val/test) con columnas:
- `reply_text` — texto del tweet respuesta (o post)
- `parent_text` — texto del padre (tweet raíz o parent inmediato)
- `label` — entero en `[0..C-1]`

Ajusta `CFG.TRAIN_CSV`, `CFG.VAL_CSV`, `CFG.TEST_CSV`.

## Flujo recomendado
1. **Precompute** (cacheo de embeddings de BERT):
   ```bash
   python Scripts/precompute_embeds.py
   ```
2. Entrenar sólo **BiLSTM + head** (BERT congelado y cacheado):
   ```bash
   python Scripts/train_pair.py
   ```
3. (Opcional) **Fine-tuning** descongelando últimas capas:
   - `CFG.FREEZE_BERT = False`
   - `CFG.UNFREEZE_LAST_N = 2`
   - `CFG.CACHE_EMBEDS = False`

## Notas sobre velocidad
- `CFG.MAX_LEN = 64`
- `CFG.BERT_MODEL = "distilbert-base-uncased"`
- `CFG.MIXED_PRECISION = True`
- `CFG.CACHE_EMBEDS = True` (Fase 1)
- `CFG.BATCH_SIZE = 64` (sube si memoria lo permite en fase cache)

¡Éxitos!
