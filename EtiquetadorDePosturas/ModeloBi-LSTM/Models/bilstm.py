# -*- coding: utf-8 -*-
import torch, torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, bidirectional, dropout, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout if num_layers>1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        feat = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(feat*2, num_classes)  # mean + max
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        avg_pool = x.mean(dim=1); max_pool, _ = x.max(dim=1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return self.classifier(self.dropout(pooled))
