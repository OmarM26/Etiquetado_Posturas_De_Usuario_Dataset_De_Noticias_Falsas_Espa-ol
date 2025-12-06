import os, torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from transformers import AutoModel
from tqdm import tqdm
from Config.config import CFG
from Data.dataset_pair import PairDataset, collate_fn
from Scripts.utils import seed_everything

def precompute_monolith(csv_path: str, split_name: str):
    ds = PairDataset(csv_path, cache_split=None)  # sin usar cache; tokeniza on-the-fly
    dl = DataLoader(
        ds, batch_size=CFG.EMBED_BATCH, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert = AutoModel.from_pretrained(CFG.BERT_MODEL).to(device)
    bert.eval()
    for p in bert.parameters():
        p.requires_grad = False

    os.makedirs(CFG.CACHE_DIR, exist_ok=True)
    out_path = os.path.join(CFG.CACHE_DIR, f"{split_name}_embeds.pt")

    rep_chunks, par_chunks = [], []
    repm_chunks, parm_chunks = [], []
    idx_total = 0
    H = None

    with torch.no_grad():
        for batch in tqdm(dl, desc=f"Precompute {split_name}"):
            # tomamos los ids/masks del batch (no hay cache aún)
            if "rep_input_ids" not in batch:
                raise RuntimeError("El DataLoader devolvió embeddings; borra cachés previos o pon cache_split=None aquí.")

            rep_ids = batch["rep_input_ids"].to(device, non_blocking=True)
            rep_msk = batch["rep_attention_mask"].to(device, non_blocking=True)
            par_ids = batch["par_input_ids"].to(device, non_blocking=True)
            par_msk = batch["par_attention_mask"].to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=CFG.MIXED_PRECISION):
                rep_last = bert(input_ids=rep_ids, attention_mask=rep_msk).last_hidden_state  # [B,T,H]
                par_last = bert(input_ids=par_ids, attention_mask=par_msk).last_hidden_state  # [B,T,H]

            rep_last = rep_last.float().cpu()
            par_last = par_last.float().cpu()
            rep_msk_cpu = rep_msk.float().cpu()
            par_msk_cpu = par_msk.float().cpu()

            if H is None:
                H = rep_last.shape[-1]

            rep_chunks.append(rep_last)      # [B,T,H]
            par_chunks.append(par_last)
            repm_chunks.append(rep_msk_cpu)  # [B,T]
            parm_chunks.append(par_msk_cpu)
            idx_total += rep_last.size(0)

    # Concatenar todo en un solo tensor por split
    rep_all  = torch.cat(rep_chunks,  dim=0)  # [N,T,H]
    par_all  = torch.cat(par_chunks,  dim=0)  # [N,T,H]
    repm_all = torch.cat(repm_chunks, dim=0)  # [N,T]
    parm_all = torch.cat(parm_chunks, dim=0)  # [N,T]

    torch.save(
        {"rep": rep_all, "par": par_all, "rep_mask": repm_all, "par_mask": parm_all},
        out_path
    )
    print(f"✅ Guardado {split_name}: {out_path}  -> shapes rep={tuple(rep_all.shape)} par={tuple(par_all.shape)}")

def main():
    seed_everything()
    assert CFG.CACHE_EMBEDS, "Activa CFG.CACHE_EMBEDS=True en Config/config.py"
    assert CFG.CACHE_FORMAT == "monolith", "Pon CACHE_FORMAT='monolith' para usar este script."

    # Recomiendo borrar cachés viejos:
    #   Remove-Item Data\\cache_monolith -Recurse -Force
    # si cambiaste BERT_MODEL/MAX_LEN/textos.

    precompute_monolith(CFG.TRAIN_CSV, "train")
    precompute_monolith(CFG.VAL_CSV,   "val")
    if os.path.exists(CFG.TEST_CSV):
        precompute_monolith(CFG.TEST_CSV, "test")

if __name__ == "__main__":
    main()
