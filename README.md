# Task 14 — Next Word Prediction (LSTM)

A next-word-prediction model trained on ~6,500 Medium article titles
(`medium_data.csv`), plus a blog post explaining the N-gram language
model that motivates the neural approach.

📝 **Medium blog post (N-gram model):** [ADD YOUR MEDIUM LINK HERE]

## 📁 Files

```
next_word_prediction/
├── medium_data.csv          # dataset (article metadata; we use the `title` column)
├── preprocess.py            # cleaning, tokenizing, vocab, n-gram sequence generation
├── model.py                 # PyTorch LSTM model definition
├── train.py                 # training loop, loss/accuracy curves, checkpoint
├── generate.py               # evaluation + sample next-word predictions & text generation
├── requirements.txt
├── data/                     # created by preprocess.py
│   ├── vocab.json
│   ├── train_sequences.npz
│   ├── val_sequences.npz
│   └── stats.json
└── outputs/                  # created by train.py / generate.py
    ├── model.pt
    ├── training_history.json
    ├── loss_curve.png
    └── sample_predictions.txt
```

## 1️⃣ Blog Post — N-Gram Model

See the Medium link at the top of this README. The post covers what an
N-gram is, how N-gram models are trained (counting + smoothing), where
they're used, their advantages (speed, interpretability, strong
baselines), and their limitations (no long-range context, data
sparsity, no notion of word similarity) — and ties it back to why
neural models like the LSTM in this project are an improvement.

## 2️⃣ Data Preparation

`preprocess.py` builds the training data from the `title` column of
`medium_data.csv` (6,508 titles):

1. **Clean & lowercase**: strip everything except letters, digits,
   apostrophes and spaces.
2. **Tokenize** on whitespace.
3. **Split at the title level** (80% train / 20% val) *before* generating
   n-gram examples, so fragments of the same title can't leak between
   train and validation.
4. **Build a vocabulary** from the training titles only (words appearing
   at least twice; everything else becomes `<UNK>`), so the validation
   set is a fair test of generalization.
5. **Generate n-gram examples**: every prefix of a title becomes one
   training example, labelled with the word that comes next (the same
   idea the blog post describes for N-gram models, just consumed by a
   neural network instead of a frequency table). Prefixes are
   pre-padded with `<PAD>` to a fixed length of 19 tokens.

```bash
python preprocess.py
```

**Actual stats from this dataset** (verified by running the script):

| | |
|---|---:|
| Titles with ≥2 tokens | 6,495 |
| Train / val titles | 5,196 / 1,299 |
| Vocabulary size (min freq = 2) | 3,124 |
| Train examples | 38,846 |
| Val examples | 9,749 |
| Max sequence length | 19 tokens |

## 3️⃣ Model Development

`model.py` defines a small LSTM language model:

```
token ids -> Embedding(vocab_size, 100) -> LSTM(100 -> 150) -> Dropout(0.2) -> Linear(150 -> vocab_size)
```

Since training sequences are pre-padded (padding comes *before* the real
tokens), the LSTM's last timestep is always the real final token of the
prefix, so the model just reads `output[:, -1, :]` as the prefix's
representation before the final classification layer.

`train.py`:
- Batches the pre-generated sequences with a PyTorch `DataLoader`
  (batch size 128).
- Trains with `CrossEntropyLoss` + Adam (lr = 1e-3) for 15 epochs.
- Tracks train/val loss, accuracy, and perplexity every epoch.
- Saves the best checkpoint (lowest val loss) to `outputs/model.pt`.
- Saves `outputs/loss_curve.png` (loss + accuracy curves) and
  `outputs/training_history.json`.

Hyperparameters (`EMBED_DIM`, `HIDDEN_DIM`, `NUM_LAYERS`, `DROPOUT`,
`EPOCHS`, `LR`, `BATCH_SIZE`) are all constants at the top of `train.py`
— worth experimenting with, per the task's step 3.

```bash
python preprocess.py   # once
python train.py
```

## 4️⃣ Evaluation & Sample Predictions

`generate.py` loads the trained checkpoint and:
- Recomputes final validation loss, accuracy, and **perplexity**
  (`exp(cross-entropy loss)` — the standard language-modelling metric
  the task asks for).
- Prints the top-5 next-word predictions (with probabilities) for five
  seed phrases.
- Greedily generates an 8-word continuation from each seed phrase, to
  show the model actually producing text, not just single-word guesses.

```bash
python generate.py
```

Output is written to `outputs/sample_predictions.txt`.

## ⚠️ About the numbers in this README

I ran `preprocess.py` for real against your uploaded `medium_data.csv`
— the data-prep stats above are the actual output of that run, and the
sandbox that produced them has no internet access, no GPU, and doesn't
have PyTorch installed, so I could not run `train.py` or `generate.py`
here. Run those two scripts yourself (locally, in Colab, or wherever you
have PyTorch), then:

- Fill in the actual final validation loss/accuracy/perplexity numbers
  from the printed output (or `outputs/training_history.json`) into this
  README before you submit.
- Include `outputs/loss_curve.png` and `outputs/sample_predictions.txt`
  in the repo as your evaluation results and sample predictions.

15 epochs on ~39,000 short training examples with this model size should
train quickly even on CPU (a few minutes), so this is a good one to
actually run end-to-end rather than trust blindly.

## ⚙️ Setup

```bash
pip install -r requirements.txt
python preprocess.py
python train.py
python generate.py
```

## Notes on scope

The dataset here is Medium article *titles* rather than full article
bodies — titles are short, clean, and plentiful (6,500+), which keeps
training fast and the vocabulary manageable, while still giving the
model plenty of natural English phrase structure to learn from (e.g.
"a beginner's guide to ...", "introduction to ...", "how to build a
..."). If you'd rather train on longer-form text, the same
`preprocess.py` pipeline works on any text column — just swap out which
column of `medium_data.csv` (or another file) it reads.
