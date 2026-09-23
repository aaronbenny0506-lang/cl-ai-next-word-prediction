"""
Hyperparameter experiments (task step 3). Trains several configs for a
few epochs each and writes outputs/experiments.json + experiments.md.
Run after preprocess.py:  python experiments.py
"""
import json, math
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from model import NextWordLSTM
from train import load_split, run_epoch, DATA_DIR, OUTPUT_DIR, BATCH_SIZE

CONFIGS = [
    dict(name="baseline",            embed=100, hidden=150, layers=1, dropout=0.2),
    dict(name="dropout 0.5",         embed=100, hidden=150, layers=1, dropout=0.5),
    dict(name="bigger (256/256)",    embed=256, hidden=256, layers=1, dropout=0.5),
    dict(name="2 layers",            embed=100, hidden=150, layers=2, dropout=0.5),
    dict(name="small (50/100)",      embed=50,  hidden=100, layers=1, dropout=0.3),
]
EPOCHS = 12

def main():
    torch.manual_seed(42)
    vocab_size = len(json.load(open(DATA_DIR / "vocab.json"))["word2id"])
    tr = DataLoader(load_split("train"), batch_size=BATCH_SIZE, shuffle=True)
    va = DataLoader(load_split("val"), batch_size=BATCH_SIZE)
    results = []
    for c in CONFIGS:
        m = NextWordLSTM(vocab_size, c["embed"], c["hidden"], c["layers"], c["dropout"])
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        best = (float("inf"), 0, 0)
        for ep in range(1, EPOCHS + 1):
            run_epoch(m, tr, crit, opt, "cpu", True)
            l, a, p = run_epoch(m, va, crit, opt, "cpu", False)
            if l < best[0]: best = (l, a, ep)
        r = dict(c, val_loss=best[0], val_acc=best[1], val_ppl=math.exp(best[0]), best_epoch=best[2])
        results.append(r); print(r)
    json.dump(results, open(OUTPUT_DIR / "experiments.json", "w"), indent=2)
    lines = ["| Config | Embed | Hidden | Layers | Dropout | Best epoch | Val loss | Val acc | Val perplexity |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['embed']} | {r['hidden']} | {r['layers']} | {r['dropout']} | {r['best_epoch']} | {r['val_loss']:.3f} | {r['val_acc']:.3f} | {r['val_ppl']:.1f} |")
    (OUTPUT_DIR / "experiments.md").write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
