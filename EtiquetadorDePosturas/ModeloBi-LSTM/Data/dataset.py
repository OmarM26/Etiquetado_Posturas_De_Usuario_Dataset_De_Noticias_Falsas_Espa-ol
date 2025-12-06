# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit

from .preprocess import simple_tok

def read_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_partitions(root: Path):
    parts = {}
    for p in ['train', 'dev', 'test', 'training', 'development']:
        d = root / p
        if d.exists() and d.is_dir():
            parts[p] = d
    return parts if parts else {'all': root}

def iter_threads(part_dir: Path):
    for event in sorted([d for d in part_dir.iterdir() if d.is_dir()]):
        for thread in sorted([d for d in event.iterdir() if d.is_dir()]):
            if (thread / 'structure.json').exists():
                yield event.name, thread

def load_thread_examples(thread_dir: Path, context: str = 'source', include_source: bool = False):
    struct = read_json(thread_dir / 'structure.json')
    source_ids = list(struct.keys())
    if len(source_ids) != 1:
        return []
    source_id = source_ids[0]
    thread_id = thread_dir.name

    def load_tweet_and_label(tid: str):
        if tid == source_id:
            tpath = thread_dir / 'source-tweet' / f'{tid}.json'
            apath = thread_dir / 'source-tweet' / 'annotation.json'
        else:
            tpath = thread_dir / 'replies' / f'{tid}.json'
            apath = thread_dir / 'replies' / tid / 'annotation.json'
        if not tpath.exists():
            return None, None
        tweet = read_json(tpath)
        text = tweet.get('text') or tweet.get('body') or ''
        label = None
        if apath.exists():
            ann = read_json(apath)
            label = ann.get('support') or ann.get('stance') or ann.get('label')
            if isinstance(label, dict):
                label = label.get('value')
            if label:
                label = label.lower().strip()
        return text, label

    parent_of = {}
    def dfs(node, parent=None):
        if isinstance(node, dict):
            for k, v in node.items():
                if parent is not None:
                    parent_of[k] = parent
                dfs(v, k)
        elif isinstance(node, list):
            for it in node:
                dfs(it, parent)
    dfs(struct)

    text_cache = {}
    def get_text(tid):
        if tid not in text_cache:
            t, _ = load_tweet_and_label(tid)
            text_cache[tid] = t or ''
        return text_cache[tid]

    LABELS = ['support', 'deny', 'query', 'comment']
    LABEL2ID = {l:i for i,l in enumerate(LABELS)}

    examples = []
    if include_source:
        src_text, src_label = load_tweet_and_label(source_id)
        if src_label in LABEL2ID and src_text:
            combined = (f"{src_text} <sep> {src_text}") if context == 'source' else src_text
            examples.append({'text': combined, 'label_id': LABEL2ID[src_label], 'tweet_id': source_id, 'thread_id': thread_id})

    replies_dir = thread_dir / 'replies'
    if replies_dir.exists():
        for f in replies_dir.glob('*.json'):
            tid = f.stem
            text, label = load_tweet_and_label(tid)
            if label not in LABEL2ID or not text:
                continue
            if context == 'none':
                combined = text
            elif context == 'parent':
                parent_id = parent_of.get(tid)
                ptext = get_text(parent_id) if parent_id else ''
                combined = f"{ptext} <sep> {text}" if ptext else text
            else:
                stext = get_text(source_id)
                combined = f"{stext} <sep> {text}" if stext else text
            examples.append({'text': combined, 'label_id': LABEL2ID[label], 'tweet_id': tid, 'thread_id': thread_id})
    return examples

def build_examples(data_root: Path, context: str, include_source: bool):
    parts = find_partitions(data_root)
    datasets = {}
    for pname, pdir in parts.items():
        all_ex = []
        groups = []
        for event, thread_dir in iter_threads(pdir):
            exs = load_thread_examples(thread_dir, context=context, include_source=include_source)
            for ex in exs:
                ex['event'] = event
                all_ex.append(ex)
                groups.append(ex['thread_id'])
        datasets[pname] = (all_ex, groups)
        print(f"[INFO] Partición {pname}: {len(all_ex)} ejemplos")
    return datasets

def split_by_threads(examples, groups, train_size=0.8, val_size=0.1, seed=42):
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    idx = np.arange(len(examples))
    train_idx, temp_idx = next(gss.split(idx, groups=groups))
    rem = 1.0 - train_size
    val_ratio = val_size / rem
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_ratio, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_idx, groups=np.array(groups)[temp_idx]))
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]
    return train_idx, val_idx, test_idx

class StanceDataset(Dataset):
    def __init__(self, examples, vocab, seq_len: int = 100):
        self.examples = examples
        self.vocab = vocab
        self.seq_len = seq_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        toks = simple_tok(ex['text'])[: self.seq_len - 2]
        toks = ['<bos>'] + toks + ['<eos>']
        ids = self.vocab.encode(toks)
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(ex['label_id'], dtype=torch.long)
        tid = ex['tweet_id']
        gid = ex['thread_id']
        return x, y, tid, gid

def pad_collate(batch):
    xs, ys, tids, gids = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    maxlen = max(lengths).item()
    padded = torch.full((len(xs), maxlen), fill_value=0, dtype=torch.long)
    for i, x in enumerate(xs):
        padded[i, : len(x)] = x
    ys = torch.stack(ys)
    return padded, lengths, ys, tids, gids
