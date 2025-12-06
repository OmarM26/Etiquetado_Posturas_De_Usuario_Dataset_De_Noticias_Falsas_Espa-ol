import os, json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report




# --- 1. CONFIGURACIÓN ---

RUTA_MODELO = "Z:/modelos_nuevos/modelo_final_contexto_postmod256_5e5_0.7317"
ARCHIVO_DATOS = "datos_etiquetados_contexto.csv"

print(f"Cargando modelo y tokenizador desde: {RUTA_MODELO}")
try:
    tokenizer = AutoTokenizer.from_pretrained(RUTA_MODELO)
    model = AutoModelForSequenceClassification.from_pretrained(RUTA_MODELO)
except OSError:
    print(f"Error: No se encontró un modelo en la carpeta '{RUTA_MODELO}'.")
    exit()

print(f"Cargando datos desde: {ARCHIVO_DATOS}")
try:
    df = pd.read_csv(ARCHIVO_DATOS)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{ARCHIVO_DATOS}'.")
    exit()

df.dropna(subset=['texto_respuesta', 'texto_padre', 'texto_raiz'], inplace=True)

# --- 2. PREPARAR DATASETS (train y val) CON CONTEXTO ---
labels = sorted(df['postura'].unique().tolist())
label2id = {label: i for i, label in enumerate(labels)}
df['labels'] = df['postura'].map(label2id)

df_train, df_val = train_test_split(
    df, test_size=200, random_state=42, stratify=df['labels']
)

print(f"\nSe usarán {len(df_train)} registros para TRAIN y {len(df_val)} para VALIDACIÓN.")

class ContextStanceDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        texto_raiz = str(row['texto_raiz'])
        texto_padre = str(row['texto_padre'])
        texto_respuesta = str(row['texto_respuesta'])

        texto_completo = f"{texto_raiz} {self.tokenizer.sep_token} {texto_padre} {self.tokenizer.sep_token} {texto_respuesta}"

        encoding = self.tokenizer.encode_plus(
            texto_completo,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(row['labels'], dtype=torch.long)
        }

train_dataset = ContextStanceDataset(df_train, tokenizer)
val_dataset   = ContextStanceDataset(df_val, tokenizer)

# --- 3. CREAR TRAINER (sin entrenar) ---
training_args = TrainingArguments(
    output_dir='./results_eval_contexto',
    per_device_eval_batch_size=8,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
)

# --- 4. OBTENER eval_loss EN VALIDACIÓN ---
print("\nEvaluando en VALIDACIÓN para obtener eval_loss...")
metrics_val = trainer.evaluate(eval_dataset=val_dataset)  # devuelve 'eval_loss'
eval_loss = metrics_val.get('eval_loss', None)

# --- 5. OBTENER train_loss (dos caminos) ---

# 5.A Intentar leer el train_loss histórico desde trainer_state.json (si existe)
train_loss_historico = None
state_path = os.path.join(RUTA_MODELO, "trainer_state.json")
if os.path.exists(state_path):
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        # Buscar el último 'loss' registrado en log_history
        log_hist = state.get("log_history", [])
        train_losses = [h["loss"] for h in log_hist if isinstance(h, dict) and "loss" in h]
        if len(train_losses) > 0:
            train_loss_historico = train_losses[-1]
    except Exception as e:
        print(f"Advertencia: no se pudo leer trainer_state.json ({e}).")

# 5.B Si no existe histórico, estimar pérdida sobre el TRAIN set (reevaluación)
train_loss_reevaluado = None
if train_loss_historico is None:
    print("Reevaluando en TRAIN para estimar 'train_loss'...")
    metrics_train = trainer.evaluate(eval_dataset=train_dataset)  # también devuelve 'eval_loss'
    train_loss_reevaluado = metrics_train.get('eval_loss', None)

# --- 6. PREDICCIONES Y MÉTRICAS CLÁSICAS EN VALIDACIÓN ---
print("\nRealizando predicciones sobre el conjunto de validación...")
predictions = trainer.predict(val_dataset)
preds = predictions.predictions.argmax(-1)
y_true = predictions.label_ids

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, preds, average='macro', zero_division=0)
accuracy = accuracy_score(y_true, preds)

# --- 7. MOSTRAR RESULTADOS ---
print("\n--- Resultados (Modelo con Contexto, ya entrenado) ---")

# Train loss
if train_loss_historico is not None:
    print(f"Train Loss (histórico del entrenamiento): {train_loss_historico:.4f}")
elif train_loss_reevaluado is not None:
    print(f"Train Loss (reevaluado sobre TRAIN con el modelo final): {train_loss_reevaluado:.4f}")
else:
    print("Train Loss: no disponible (no hay trainer_state.json y no se pudo reevaluar).")

# Eval loss
if eval_loss is not None:
    print(f"Eval Loss (validación): {eval_loss:.4f}")
else:
    # fallback desde predict: test_loss
    test_loss = predictions.metrics.get('test_loss', None)
    if test_loss is not None:
        print(f"Eval Loss (desde predict/test_loss): {test_loss:.4f}")

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1-Score: {f1_macro:.4f}")
print("-" * 50)
print("Informe de Clasificación Detallado (F1-Score por clase):")
target_names = sorted(label2id, key=label2id.get)
print(classification_report(y_true, preds, target_names=target_names, zero_division=0))
