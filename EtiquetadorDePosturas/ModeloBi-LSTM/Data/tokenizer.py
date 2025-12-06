from transformers import AutoTokenizer
from Config.config import CFG

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(CFG.BERT_MODEL, use_fast=True)
    return _tokenizer
