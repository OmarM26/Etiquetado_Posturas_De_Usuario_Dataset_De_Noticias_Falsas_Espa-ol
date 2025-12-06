# -*- coding: utf-8 -*-
import json
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds, golds, tids_all, gids_all = [], [], [], [], []
    with torch.no_grad():
        for x, lengths, y, tids, gids in loader:
            x = x.to(device); y = y.to(device); lengths = lengths.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            losses.append(loss.item())
            preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
            golds.extend(y.detach().cpu().tolist())
            tids_all.extend(list(tids))
            gids_all.extend(list(gids))
    f1 = f1_score(golds, preds, average='macro')
    return float(np.mean(losses)), float(f1), preds, golds, tids_all, gids_all

def save_report(golds, preds, labels, out_json):
    rep = classification_report(golds, preds, target_names=labels, digits=4, output_dict=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    return rep

def save_predictions(tids, gids, golds, preds, id2label, out_json):
    rows = []
    for tid, gid, y_true, y_pred in zip(tids, gids, golds, preds):
        rows.append({"tweet_id": tid, "thread_id": gid, "gold": id2label[y_true], "pred": id2label[y_pred]})
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
