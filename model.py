"""
LSTM next-word-prediction model (PyTorch).

Architecture:
    token ids -> Embedding -> LSTM -> take last timestep's hidden state
              -> Dropout -> Linear -> logits over the vocabulary

Because training sequences are PRE-padded (padding tokens come first,
real tokens are right-aligned - preprocess.py), the last timestep of
the LSTM's output is always the real final token of the prefix, so we
can just take `output[:, -1, :]` without any pack_padded_sequence
bookkeeping.
"""

import torch
import torch.nn as nn


class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 100,
                 hidden_dim: int = 150, num_layers: int = 1,
                 dropout: float = 0.2, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) token ids -> logits: (batch, vocab_size)"""
        emb = self.embedding(x)                 # (B, T, E)
        out, _ = self.lstm(emb)                  # (B, T, H)
        last = out[:, -1, :]                      # (B, H) — real last token (pre-padded input)
        last = self.dropout(last)
        logits = self.fc(last)                    # (B, V)
        return logits
