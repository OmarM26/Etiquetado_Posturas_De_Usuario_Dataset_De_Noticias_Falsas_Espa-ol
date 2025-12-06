from dataclasses import dataclass
from datetime import datetime

@dataclass
class CFG:
    TEXT_COL_REPLY = "reply"
    TEXT_COL_PARENT = "parent"
    TEXT_COL_ROOT = "root"
    LABEL_COL = "label_sdqc"

    OUT_DIR = "C:/Users/omarm/Downloads/estado del arte papers/GitHub2/outputs"
    BEST_BASENAME = "best_model.pt"
    CKPT_PATH = None

    # Rutas
    TRAIN_CSV: str = "Data/rumoureval_train.csv"
    VAL_CSV: str   = "Data/rumoureval_dev.csv"
    TEST_CSV: str  = "Data/rumoureval_dev.csv"

    # Columnas
    TEXT_COL_REPLY: str  = "text"
    TEXT_COL_PARENT: str = "context_text"
    LABEL_COL: str       = "label_sdqc"

    # --- ETIQUETAS CORREGIDAS ---
    LABELS = ["comment", "support", "query", "deny"]
    LABEL2ID = {'comment': 0, 'support': 1, 'query': 2, 'deny': 3}
    TARGET_NAMES = ['comment','support','query','deny']
    NUM_CLASSES: int = 4

    # --- MODELO / TOKENIZACIÓN ---
    BERT_MODEL: str = "distilbert-base-uncased"
    MAX_LEN: int = 128
    FREEZE_BERT: bool = False # Fine-tuning ACTIVADO
    UNFREEZE_LAST_N: int = 2  # Descongelar la última capa
    
    CACHE_EMBEDS: bool = False # Desactivar caché para fine-tuning
    
    # --- BiLSTM ---
    LSTM_HIDDEN: int = 128
    LSTM_LAYERS: int = 2       # CAMBIADO A 2 para que el dropout funcione
    LSTM_DROPOUT: float = 0.4
    BIDIRECTIONAL: bool = True

    # --- ENTRENAMIENTO ---
    BATCH_SIZE: int = 32 # Reducir un poco por si aumenta el uso de memoria
    EVAL_BS: int = 64
    EPOCHS: int = 20
    LR_HEAD: float = 5e-5
    LR_BERT: float = 2e-5
    WEIGHT_DECAY: float = 1e-2
    PATIENCE: int = 3
    SEED: int = 42

    # Loss
    LOSS: str = "focal" # Puede ser "ce" o "focal"
    FOCAL_GAMMA: float = 2.0
    LABEL_SMOOTH: float = 0.05

    # Dataloader
    NUM_WORKERS: int = 0