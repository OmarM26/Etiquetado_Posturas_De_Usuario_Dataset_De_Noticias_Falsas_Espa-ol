# -*- coding: utf-8 -*-
import re
from collections import Counter

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
USER_RE = re.compile(r"@[A-Za-z0-9_]{1,15}")
HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")

TOKEN_RE = re.compile(r"<url>|<user>|<hashtag>|[\w']+|[^\w\s]", re.UNICODE)

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "[SEP]", "<url>", "<user>", "<hashtag>"]


def normalize_tweet(text: str) -> str:
    text = (text or "").replace("\n", " ").strip()
    text = URL_RE.sub("<url>", text)
    text = USER_RE.sub("<user>", text)
    text = HASHTAG_RE.sub("<hashtag>", text)
    return text

def simple_tok(text: str):
    text = normalize_tweet(text)
    return TOKEN_RE.findall(text.lower())

class Vocab:
    def __init__(self, min_freq=2, max_size=50000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.itos = []
        self.stoi = {}

    def build(self, token_lists):
        freq = Counter()
        for toks in token_lists:
            freq.update(toks)
        self.itos = list(SPECIAL_TOKENS)
        words = [w for w, c in freq.items() if c >= self.min_freq and w not in SPECIAL_TOKENS]
        words.sort(key=lambda w: (-freq[w], w))
        if self.max_size:
            words = words[: max(0, self.max_size - len(self.itos))]
        self.itos += words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        return [self.stoi.get(t, self.stoi["<unk>"]) for t in toks]
