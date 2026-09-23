"""
Train the LSTM next-word-prediction model on the sequences produced by
preprocess.py.

Run:
    python preprocess.py     # once, to build ./data/
    python train.py

Produces:
    outputs/model.pt               trained model weights + vocab size
    outputs/training_history.json  per-epoch train/val loss, accuracy, perplexity
    outputs/loss_curve.png         training/validation loss and accuracy curves
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import NextWordLSTM

RANDOM_STATE = 42
BATCH_SIZE = 128
EMBED_DIM = 256
HIDDEN_DIM = 256
NUM_LAYERS = 1
DROPOUT = 0.5
EPOCHS = 10
LR = 1e-3

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_split(name: str):
    npz = np.load(DATA_DIR / f"{name}_sequences.npz")
    X = torch.from_numpy(npz["X"]).long()
    y = torch.from_numpy(npz["y"]).long()
    return TensorDataset(X, y)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * X.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_n += X.size(0)
    avg_loss = total_loss / total_n
    accuracy = total_correct / total_n
    perplexity = math.exp(avg_loss)
    return avg_loss, accuracy, perplexity


def main():
    torch.manual_seed(RANDOM_STATE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(DATA_DIR / "vocab.json") as f:
        vocab = json.load(f)
    vocab_size = len(vocab["word2id"])
    print(f"Vocab size: {vocab_size}")

    train_ds = load_split("train")
    val_ds = load_split("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = NextWordLSTM(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "train_ppl": [], "val_ppl": []}

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["train_ppl"].append(tr_ppl)
        history["val_ppl"].append(va_ppl)

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} ppl {tr_ppl:7.2f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} ppl {va_ppl:7.2f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "max_len": vocab["max_len"],
            }, OUTPUT_DIR / "model.pt")

    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ---- loss / accuracy curves ----
    import matplotlib.pyplot as plt
    epochs_range = range(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs_range, history["train_loss"], label="train")
    axes[0].plot(epochs_range, history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(epochs_range, history["train_acc"], label="train")
    axes[1].plot(epochs_range, history["val_acc"], label="val")
    axes[1].set_title("Next-word accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
    plt.close()

    print(f"\nBest val loss: {best_val_loss:.4f} (perplexity {math.exp(best_val_loss):.2f})")
    print(f"Saved outputs/model.pt, outputs/training_history.json, outputs/loss_curve.png")


if __name__ == "__main__":
    main()
