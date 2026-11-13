"""From-scratch BiLSTM: tokenizer, vocabulary, model and training loop.

No pretrained weights. train_bilstm.py uses this for the 3-seed run, and the
step 5 tuning can reuse it by changing Config.
"""
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

import common

PAD, UNK = 0, 1

# Words, keeping contractions whole so "didn't" stays one token that carries its
# negation, plus ! and ? because they can signal strong feeling. Other punctuation goes.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?]")


@dataclass
class Config:
    embed_dim: int = 128
    hidden: int = 128  # per direction, so each time step outputs 2 x hidden
    dropout: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_len: int = 200
    min_freq: int = 2  # rarer training words become <unk>
    max_epochs: int = 20
    patience: int = 4  # stop after this many epochs with no macro-F1 improvement
    grad_clip: float = 1.0  # stops exploding gradients, a known RNN problem
    class_weighting: bool = True


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(texts, min_freq):
    """Pass training texts only: words unseen in training must map to <unk>."""
    counts = Counter(token for text in texts for token in tokenize(text))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    # Ties sort alphabetically, so the same data always gives the same vocabulary.
    for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode(text, vocab, max_len):
    """Keep the first max_len tokens. Every review has at least one token slot."""
    ids = [vocab.get(token, UNK) for token in tokenize(text)[:max_len]]
    return ids or [UNK]


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.items = [encode(text, vocab, max_len) for text in texts]
        self.labels = list(labels)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index], self.labels[index]


def collate(batch):
    """Pad to the longest review in this batch, which saves time over a fixed length."""
    longest = max(len(ids) for ids, _ in batch)
    padded = [ids + [PAD] * (longest - len(ids)) for ids, _ in batch]
    return torch.tensor(padded), torch.tensor([label for _, label in batch])


class BiLSTMClassifier(nn.Module):
    """ids (B, T) -> embed (B, T, E) -> BiLSTM (B, T, 2H) -> pool (B, 4H) -> logits (B, 2)"""

    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(cfg.embed_dim, cfg.hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(4 * cfg.hidden, 2)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.1)
        self.embed.weight.data[PAD].zero_()
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)  # keeps signals stable across time steps
            elif "bias" in name:
                nn.init.zeros_(param)
                # Forget-gate bias of 1: the LSTM starts out remembering by default.
                # PyTorch adds bias_ih and bias_hh together, so set only one of them.
                if "bias_ih" in name:
                    size = param.size(0)
                    param.data[size // 4 : size // 2].fill_(1.0)

    def forward(self, ids):
        lengths = (ids != PAD).sum(dim=1).cpu()
        # Packing makes the LSTM stop at each review's true end, so the backward
        # pass starts at the last real word instead of at padding.
        packed = pack_padded_sequence(self.embed(ids), lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=ids.size(1))

        # Mean pooling summarises the whole review. Max pooling keeps the strongest
        # signal, so one "disappointed" is not diluted in a long review. Both must
        # ignore padding.
        mask = (ids != PAD).unsqueeze(-1)
        mean = (output * mask).sum(dim=1) / lengths.to(ids.device).unsqueeze(-1)
        maximum = output.masked_fill(~mask, -1e4).max(dim=1).values
        return self.head(self.dropout(torch.cat([mean, maximum], dim=1)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_weights(labels, enabled, device):
    """n / (2 * n_class): the same formula as scikit-learn's class_weight='balanced'."""
    if not enabled:
        return torch.ones(2, device=device)
    counts = np.bincount(labels, minlength=2)
    return torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)


@torch.no_grad()
def predict(model, loader, loss_fn, device):
    model.eval()
    predictions, total_loss, seen = [], 0.0, 0
    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        with torch.amp.autocast("cuda"):
            logits = model(ids)
            loss = loss_fn(logits, labels)
        predictions.append(logits.argmax(dim=1).cpu())
        total_loss += loss.item() * len(labels)
        seen += len(labels)
    return torch.cat(predictions).numpy(), total_loss / seen


def train_once(train_df, val_df, cfg, seed, verbose=True):
    """One full training run. Keeps the weights from the best validation macro-F1 epoch."""
    set_seed(seed)
    device = torch.device("cuda")
    vocab = build_vocab(train_df["reviews"], cfg.min_freq)
    train_labels = train_df["sentiments"].to_numpy()
    val_labels = val_df["sentiments"].to_numpy()

    train_loader = DataLoader(
        ReviewDataset(train_df["reviews"], train_labels.tolist(), vocab, cfg.max_len),
        batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        ReviewDataset(val_df["reviews"], val_labels.tolist(), vocab, cfg.max_len),
        batch_size=128, collate_fn=collate,
    )

    model = BiLSTMClassifier(len(vocab), cfg).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_labels, cfg.class_weighting, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda")

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    history, best_f1, best_epoch, best_state = [], -1.0, 0, None
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for ids, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = loss_fn(model(ids), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # clipping must see real gradient sizes
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
            seen += len(labels)

        pred, val_loss = predict(model, val_loader, loss_fn, device)
        metrics = common.evaluate(val_labels, pred)
        history.append({
            "seed": seed, "epoch": epoch, "train_loss": total_loss / seen, "val_loss": val_loss,
            **{f"val_{key}": value for key, value in metrics.items()},
        })
        if verbose:
            print(
                f"  seed {seed} epoch {epoch:>2}: train loss {total_loss / seen:.3f}, "
                f"val loss {val_loss:.3f}, acc {metrics['accuracy']:.4f}, "
                f"macro-F1 {metrics['macro_f1']:.4f}, neg-F1 {metrics['neg_f1']:.4f}"
            )
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            # Keep the copy in CPU memory, so it does not count towards GPU use.
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        elif epoch - best_epoch >= cfg.patience:
            break
    seconds = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    model.load_state_dict(best_state)
    pred, _ = predict(model, val_loader, loss_fn, device)
    return {
        "seed": seed, "metrics": common.evaluate(val_labels, pred), "pred": pred,
        "history": history, "best_epoch": best_epoch, "epochs_run": len(history),
        "seconds": seconds, "peak_gpu_gb": peak_gb, "vocab": vocab, "state": best_state,
        "n_params": sum(p.numel() for p in model.parameters()), "config": asdict(cfg),
    }
