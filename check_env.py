"""Environment check: confirms the GPU can really train both neural architectures.

Run:  .\\.venv\\Scripts\\python.exe check_env.py

It trains on a single batch of real reviews, using the settings the project
plans to use (mixed precision, weighted loss). A falling loss with no NaNs
shows the whole pipeline works. These are NOT model results: nothing here is
held out, and test.json is never opened.
"""
import json
import platform
import sys
import time
import warnings

import accelerate
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
STEPS = 20
BATCH_SIZE = 16  # the largest batch we expect to want for the transformer
MAX_LEN = 256  # the longest length in the planned 128-256 range
MODEL_NAME = "distilbert-base-uncased"


def train_steps(model, forward, labels, lr):
    """Run STEPS fp16 updates on one batch. Returns first loss, last loss, stats."""
    # Inverse-frequency class weights, the same idea as the real weighted loss.
    # Negatives are ~15% of the data, so their errors count for more.
    counts = torch.bincount(labels, minlength=2).float()
    weights = counts.sum() / (2 * counts.clamp(min=1))
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # GradScaler stops small fp16 gradients from rounding down to zero.
    scaler = torch.amp.GradScaler("cuda")

    torch.cuda.reset_peak_memory_stats()
    losses = []
    start = time.perf_counter()
    model.train()
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = loss_fn(forward(), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    if not np.isfinite(losses).all():
        raise RuntimeError(f"Loss became NaN or infinite: {losses}")
    if losses[-1] >= losses[0]:
        raise RuntimeError(f"Loss did not fall: {losses[0]:.3f} -> {losses[-1]:.3f}")
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "steps_per_sec": STEPS / seconds,
        "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "params": sum(p.numel() for p in model.parameters()),
    }


class SmokeBiLSTM(nn.Module):
    """Throwaway stand-in for step 3's model: embedding, BiLSTM, pooling, linear."""

    def __init__(self, vocab_size, embed_dim=128, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 2)

    def forward(self, ids):
        output, _ = self.lstm(self.embed(ids))
        return self.head(output.mean(dim=1))


def report(name, stats):
    print(
        f"{name:<11} loss {stats['first_loss']:.3f} -> {stats['last_loss']:.3f} | "
        f"{stats['steps_per_sec']:.1f} steps/s | peak GPU {stats['peak_gb']:.2f} GB | "
        f"{stats['params']:,} params"
    )


def main():
    print(f"Python        {platform.python_version()} ({sys.executable})")
    for module in (torch, transformers, accelerate, sklearn, pd, np, matplotlib):
        print(f"{module.__name__:<13} {module.__version__}")

    if not torch.cuda.is_available():
        sys.exit("FAILED: torch.cuda.is_available() is False, so there is no GPU training.")
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU           {torch.cuda.get_device_name(0)}, {total_gb:.1f} GB, CUDA {torch.version.cuda}")

    with open("train.json", encoding="utf-8") as handle:
        rows = json.load(handle)
    negatives = [r for r in rows if r["sentiments"] == 0][: BATCH_SIZE // 2]
    positives = [r for r in rows if r["sentiments"] == 1][: BATCH_SIZE // 2]
    batch = negatives + positives
    texts = [r["reviews"] for r in batch]
    labels = torch.tensor([r["sentiments"] for r in batch], device=device)
    print(f"\nTraining on one batch of {len(batch)} real reviews, padded to {MAX_LEN} tokens, {STEPS} steps\n")

    # Pad every review to full length so memory reflects the worst case.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoded = tokenizer(
        texts, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt"
    ).to(device)

    lstm = SmokeBiLSTM(tokenizer.vocab_size).to(device)
    report("BiLSTM", train_steps(lstm, lambda: lstm(encoded["input_ids"]), labels, lr=1e-3))
    del lstm
    torch.cuda.empty_cache()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bert = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    report("DistilBERT", train_steps(bert, lambda: bert(**encoded).logits, labels, lr=2e-5))

    print("\nPASSED: both models trained on the GPU with fp16 and their loss fell.")


if __name__ == "__main__":
    main()
