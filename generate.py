"""
Load the trained model and:
  1. report final validation accuracy / perplexity,
  2. generate sample next-word predictions and short continuations from
     a handful of seed phrases, to demonstrate the model qualitatively.

Run (after train.py):
    python generate.py
Produces:
    outputs/sample_predictions.txt
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from model import NextWordLSTM
from preprocess import clean_text, tokenize

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

SEED_PHRASES = [
    "how to build a",
    "a beginner's guide to",
    "introduction to",
    "the future of",
    "why you should learn",
]
WORDS_TO_GENERATE = 8
TOP_K_SHOWN = 5


def load_model_and_vocab(device):
    with open(DATA_DIR / "vocab.json") as f:
        vocab = json.load(f)
    word2id = vocab["word2id"]
    id2word = {int(i): w for i, w in vocab["id2word"].items()}
    max_len = vocab["max_len"]

    ckpt = torch.load(OUTPUT_DIR / "model.pt", map_location=device)
    model = NextWordLSTM(
        ckpt["vocab_size"], ckpt["embed_dim"], ckpt["hidden_dim"],
        ckpt["num_layers"], ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, word2id, id2word, max_len


def encode_prefix(text: str, word2id: dict, max_len: int) -> torch.Tensor:
    unk = word2id["<UNK>"]
    ids = [word2id.get(t, unk) for t in tokenize(text)]
    width = max_len - 1
    ids = ids[-width:]
    padded = [0] * (width - len(ids)) + ids
    return torch.tensor([padded], dtype=torch.long)


@torch.no_grad()
def predict_next(model, text, word2id, id2word, max_len, device, top_k=TOP_K_SHOWN):
    x = encode_prefix(text, word2id, max_len).to(device)
    logits = model(x)[0]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, top_k)
    return [(id2word[i.item()], p.item()) for i, p in zip(top_ids, top_probs)]


@torch.no_grad()
def generate_continuation(model, seed, word2id, id2word, max_len, device, n_words=WORDS_TO_GENERATE):
    text = seed
    for _ in range(n_words):
        x = encode_prefix(text, word2id, max_len).to(device)
        logits = model(x)[0]
        next_id = torch.argmax(logits).item()
        next_word = id2word[next_id]
        if next_word in ("<PAD>", "<UNK>"):
            break
        text = text + " " + next_word
    return text


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, word2id, id2word, max_len = load_model_and_vocab(device)

    # ---- final val metrics (re-computed here for the report) ----
    npz = np.load(DATA_DIR / "val_sequences.npz")
    X_val = torch.from_numpy(npz["X"]).long()
    y_val = torch.from_numpy(npz["y"]).long()
    with torch.no_grad():
        logits = model(X_val.to(device))
        loss = torch.nn.functional.cross_entropy(logits, y_val.to(device)).item()
        acc = (logits.argmax(dim=-1).cpu() == y_val).float().mean().item()
    ppl = math.exp(loss)

    lines = []
    lines.append(f"Final validation loss:       {loss:.4f}")
    lines.append(f"Final validation accuracy:   {acc:.4f}")
    lines.append(f"Final validation perplexity: {ppl:.2f}")
    lines.append("")
    lines.append("=== Top-k next-word predictions for seed phrases ===")
    for seed in SEED_PHRASES:
        preds = predict_next(model, seed, word2id, id2word, max_len, device)
        pred_str = ", ".join(f"{w} ({p:.2f})" for w, p in preds)
        lines.append(f'  "{seed}" -> {pred_str}')
    lines.append("")
    lines.append("=== Greedy multi-word continuations from seed phrases ===")
    for seed in SEED_PHRASES:
        cont = generate_continuation(model, seed, word2id, id2word, max_len, device)
        lines.append(f'  "{seed}" -> "{cont}"')

    report = "\n".join(lines)
    print(report)
    (OUTPUT_DIR / "sample_predictions.txt").write_text(report + "\n")
    print("\nSaved outputs/sample_predictions.txt")


if __name__ == "__main__":
    main()
