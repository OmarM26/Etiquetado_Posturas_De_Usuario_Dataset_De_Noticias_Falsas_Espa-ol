import torch, pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from Config.config import CFG
from Data.dataset_pair import PairDataset, collate_fn
from Models.bilstm_pair import BiLSTMPair
from Scripts.utils import to_device

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(PairDataset(CFG.TEST_CSV, cache_split=("test" if CFG.CACHE_EMBEDS else None)),
                             batch_size=CFG.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    model = BiLSTMPair().to(device)
    state = torch.load(f"{CFG.OUT_DIR}/{CFG.BEST_BASENAME}", map_location=device)
    model.load_state_dict(state)
    model.eval()

    y_true, y_pred = [], []
    for batch in test_loader:
        batch = to_device(batch, device)
        logits = model(batch)
        y_true.extend(batch["label"].cpu().numpy().tolist())
        y_pred.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())

    print(classification_report(y_true, y_pred, digits=4))
    print("Macro-F1:", f1_score(y_true, y_pred, average="macro"))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
