import torch
import torch.nn as nn
from transformers import AutoModel
from Config.config import CFG

class SeqEncoder(nn.Module):
    def __init__(self, hidden_bert=768):
        super().__init__()
        self.bert = AutoModel.from_pretrained(CFG.BERT_MODEL)

        for param in self.bert.parameters():
            param.requires_grad = False

        if not CFG.FREEZE_BERT and CFG.UNFREEZE_LAST_N > 0:
            num_layers = len(self.bert.transformer.layer)
            layers_to_unfreeze = self.bert.transformer.layer[num_layers - CFG.UNFREEZE_LAST_N:]
            
            print(f"\nDescongelando las últimas {len(layers_to_unfreeze)} capas de DistilBERT...")
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
        
        self.lstm = nn.LSTM(
            input_size=hidden_bert,
            hidden_size=CFG.LSTM_HIDDEN,
            num_layers=CFG.LSTM_LAYERS,
            dropout=CFG.LSTM_DROPOUT,
            bidirectional=CFG.BIDIRECTIONAL,
            batch_first=True
        )

    def forward(self, ids, mask):
        bert_out = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        lstm_out, _ = self.lstm(bert_out)
        return lstm_out

class BiLSTMPair(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_bert = 768
        self.enc = SeqEncoder(hidden_bert=hidden_bert)

        enc_out_dim = CFG.LSTM_HIDDEN * (2 if CFG.BIDIRECTIONAL else 1)
        feat_dim = enc_out_dim * 4
        
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, CFG.NUM_CLASSES),
        )
        
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, batch):
        rep_ids, rep_mask = batch["rep_input_ids"], batch["rep_attention_mask"]
        par_ids, par_mask = batch["par_input_ids"], batch["par_attention_mask"]

        rep_enc = self.enc(ids=rep_ids, mask=rep_mask)
        par_enc = self.enc(ids=par_ids, mask=par_mask)

        rep_vec = rep_enc[:, 0, :]
        par_vec = par_enc[:, 0, :]

        diff = torch.abs(rep_vec - par_vec)
        prod = rep_vec * par_vec
        
        combined_features = torch.cat([rep_vec, par_vec, diff, prod], dim=1)
        
        logits = self.head(combined_features)
        return logits