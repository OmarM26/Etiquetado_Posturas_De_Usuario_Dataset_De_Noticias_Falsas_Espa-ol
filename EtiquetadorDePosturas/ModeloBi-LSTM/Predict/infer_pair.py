import torch
from Config.config import CFG
from Data.tokenizer import get_tokenizer
from Models.bilstm_pair import BiLSTMPair

@torch.no_grad()
def infer(reply_text: str, parent_text: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMPair().to(device)
    state = torch.load(f"{CFG.OUT_DIR}/{CFG.BEST_BASENAME}", map_location=device)
    model.load_state_dict(state); model.eval()

    tok = get_tokenizer()
    def tok1(x):
        t = tok([x], padding="max_length", truncation=True, max_length=CFG.MAX_LEN, return_tensors="pt")
        return t["input_ids"].to(device), t["attention_mask"].to(device)
    rep_ids, rep_mask = tok1(reply_text)
    par_ids, par_mask = tok1(parent_text)

    last = model.enc.encode_with_bert(rep_ids, rep_mask)
    last_p = model.enc.encode_with_bert(par_ids, par_mask)
    z = model.enc(last, rep_mask)
    z_p = model.enc(last_p, par_mask)
    pair = torch.cat([z, z_p, torch.abs(z - z_p), z * z_p], dim=1)
    logits = model.head(pair)
    pred = torch.argmax(logits, dim=-1).item()
    return pred, torch.softmax(logits, dim=-1).cpu().numpy().tolist()

if __name__ == "__main__":
    label, probs = infer("I doubt this is true", "Breaking: ...")
    print("label:", label, "probs:", probs)
