import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import TrainerCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss

# --- 1. CONFIGURACIÓN Y CARGA DE DATOS ---
now = datetime.now()
timestamp_str = now.strftime("%Y-%m-%d_%H-%M")
NOMBRE_MODELO =  "dccuchile/bert-base-spanish-wwm-uncased" #primeros 5 #"PlanTL-GOB-ES/roberta-base-bne" 
ARCHIVO_DATOS = "datos_etiquetados_contexto.csv"
output_dir = f"Z:/modelos_nuevos/modelo_final_contexto_postmod128_5e5_{timestamp_str}"
os.makedirs(output_dir, exist_ok=True)

try:
    df = pd.read_csv(ARCHIVO_DATOS)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{ARCHIVO_DATOS}'.")
    exit()

df.dropna(subset=['texto_respuesta', 'texto_padre', 'texto_raiz'], inplace=True)

# --- 2. PREPARACIÓN DE DATOS ---
labels = sorted(df['postura'].unique().tolist())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
df['labels'] = df['postura'].map(label2id)

#df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['labels'])
df_train, df_val = train_test_split(df, train_size=1001, test_size=200, random_state=42, stratify=df['labels'])

print(f"Tamaño del conjunto de entrenamiento: {len(df_train)}")
print(f"Tamaño del conjunto de validación: {len(df_val)}")
print("\nDistribución de clases en entrenamiento:")
print(df_train['postura'].value_counts())

tokenizer = AutoTokenizer.from_pretrained(NOMBRE_MODELO)

class ContextStanceDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df
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
            texto_completo, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(row['labels'], dtype=torch.long)
        }

train_dataset = ContextStanceDataset(df_train, tokenizer)
val_dataset = ContextStanceDataset(df_val, tokenizer)

# --- 3. CONFIGURACIÓN DEL MODELO, PESOS DE CLASE Y ENTRENAMIENTO ---
print("\nCalculando pesos de clase para manejar el desbalance...")
class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(df_train['labels']), y=df_train['labels'].to_numpy()
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")
print("Pesos calculados:", class_weights)

model = AutoModelForSequenceClassification.from_pretrained(
    NOMBRE_MODELO, 
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    use_safetensors=True
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}
    
class CustomTrainer(Trainer):
    # CORRECCIÓN: Se añade **kwargs para aceptar argumentos adicionales
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=f'./results_contexto_{timestamp_str}',
    num_train_epochs=8, per_device_train_batch_size=8, per_device_eval_batch_size=8,
    warmup_steps=500, weight_decay=0.01, logging_dir=f'./logs_contexto_{timestamp_str}',
    logging_strategy="epoch", report_to="none", save_strategy="epoch",
    eval_strategy="epoch", load_best_model_at_end=True, learning_rate=5e-5,
    metric_for_best_model="f1",      # <-- Le dices que la métrica a observar es 'f1' (nuestro Macro F1)
    greater_is_better=True  
)

class MetricsLoggerCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.metrics_history = []
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_world_process_zero:
            self.metrics_history.append({
                'epoch': state.epoch, 'eval_loss': metrics.get('eval_loss'),
                'eval_accuracy': metrics.get('eval_accuracy'), 'eval_f1': metrics.get('eval_f1')
            })

trainer = CustomTrainer(
    model=model, args=training_args, train_dataset=train_dataset,
    eval_dataset=val_dataset, compute_metrics=compute_metrics, callbacks=[MetricsLoggerCallback()],
)

# --- 4. INICIAR ENTRENAMIENTO ---
print("\nIniciando el entrenamiento del modelo con contexto y pesos de clase...")
trainer.train()

# --- 5. GENERAR GRÁFICO DE MÉTRICAS ---
print("\nEntrenamiento finalizado. Generando gráfico de métricas...")
metrics_callback = next((cb for cb in trainer.callback_handler.callbacks if isinstance(cb, MetricsLoggerCallback)), None)
if metrics_callback:
    metrics_df = pd.DataFrame(metrics_callback.metrics_history)
    if not metrics_df.empty:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 8))
        plt.plot(metrics_df['epoch'], metrics_df['eval_loss'], label='Loss', marker='o')
        plt.plot(metrics_df['epoch'], metrics_df['eval_accuracy'], label='Accuracy', marker='o')
        plt.plot(metrics_df['epoch'], metrics_df['eval_f1'], label='F1-Score (Macro)', marker='o')
        plt.title('Métricas de rendimiento por Época (con Pesos de Clase)', fontsize=16)
        plt.xlabel('Época', fontsize=12)
        plt.ylabel('Valor de la Métrica', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'metricas_entrenamiento.png')
        plt.savefig(plot_path)
        print(f"Gráfico de métricas guardado en '{plot_path}'.")
    else:
        print("No se encontraron métricas de evaluación para graficar.")
else:
    print("Error: No se encontró la instancia de MetricsLoggerCallback.")

# --- 6. GUARDAR EL MODELO FINAL ---
print("Guardando el mejor modelo final...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Modelo y tokenizador guardados en la carpeta '{output_dir}'.")