"""Fine-tuned pretrained transformer with a single linear head.

Works for DistilBERT and RoBERTa. train_transformer.py uses this for the 3-seed
run, and the step 5 tuning can reuse it by changing Config.
"""
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

import common
from bilstm import class_weights, set_seed


@dataclass
class Config:
    model_name: str = "distilbert-base-uncased"
    lr: float = 2e-5  # small: the pretrained weights only need adjusting
    epochs: int = 4
    batch_size: int = 16
    max_len: int = 256
    dropout: float = 0.1
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1  # share of all steps spent ramping the rate up
    grad_clip: float = 1.0
    class_weighting: bool = True


class TransformerClassifier(nn.Module):
    """ids (B, T) -> encoder (B, T, 768) -> first token (B, 768) -> logits (B, 2)"""

    def __init__(self, model_name, dropout):
        super().__init__()
        # RoBERTa ships an extra pooling layer we would not use, so leave it out.
        extra = {"add_pooling_layer": False} if "roberta" in model_name else {}
        self.encoder = AutoModel.from_pretrained(model_name, **extra)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # The first token ([CLS] or <s>) is the usual summary of a whole sequence
        # for classification.
        return self.head(self.dropout(states[:, 0]))


class Encoded(Dataset):
    def __init__(self, encodings, labels):
        self.encodings, self.labels = encodings, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"input_ids": self.encodings[index], "label": self.labels[index]}


def make_loader(texts, labels, tokenizer, cfg, batch_size, generator=None):
    """Truncate here and pad per batch, which is faster than always padding to max_len."""
    encodings = tokenizer(list(texts), truncation=True, max_length=cfg.max_len)["input_ids"]

    def collate(batch):
        padded = tokenizer.pad([{"input_ids": item["input_ids"]} for item in batch], return_tensors="pt")
        return padded["input_ids"], padded["attention_mask"], torch.tensor([item["label"] for item in batch])

    return DataLoader(
        Encoded(encodings, list(labels)), batch_size=batch_size, shuffle=generator is not None,
        generator=generator, collate_fn=collate,
    )


def optimizer_groups(model, weight_decay):
    """Skip weight decay on biases and normalisation weights, which are all 1-D.

    Decay discourages large weights in the main matrices. Applying it to these
    is a known way to hurt training.
    """
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def predict(model, loader, loss_fn, device):
    model.eval()
    predictions, total_loss, seen = [], 0.0, 0
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        with torch.amp.autocast("cuda"):
            logits = model(ids, mask)
            loss = loss_fn(logits, labels)
        predictions.append(logits.argmax(dim=1).cpu())
        total_loss += loss.item() * len(labels)
        seen += len(labels)
    return torch.cat(predictions).numpy(), total_loss / seen


def train_once(train_df, val_df, cfg, seed, verbose=True):
    """One full run. Keeps the weights from the best validation macro-F1 epoch."""
    set_seed(seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    train_labels = train_df["sentiments"].to_numpy()
    val_labels = val_df["sentiments"].to_numpy()
    train_loader = make_loader(
        train_df["reviews"], train_labels.tolist(), tokenizer, cfg, cfg.batch_size,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = make_loader(val_df["reviews"], val_labels.tolist(), tokenizer, cfg, batch_size=64)

    model = TransformerClassifier(cfg.model_name, cfg.dropout).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_labels, cfg.class_weighting, device))
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg.weight_decay), lr=cfg.lr)
    total_steps = cfg.epochs * len(train_loader)
    # Warmup avoids large early updates while the new head is still random; the
    # rate then falls linearly to zero, so the weights settle by the end.
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_fraction * total_steps), total_steps
    )
    scaler = torch.amp.GradScaler("cuda")

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    history, best_f1, best_epoch, best_state = [], -1.0, 0, None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = loss_fn(model(ids, mask), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:  # a lower scale means the step was skipped
                scheduler.step()
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
                f"  seed {seed} epoch {epoch}: train loss {total_loss / seen:.3f}, "
                f"val loss {val_loss:.3f}, acc {metrics['accuracy']:.4f}, "
                f"macro-F1 {metrics['macro_f1']:.4f}, neg-F1 {metrics['neg_f1']:.4f} "
                f"[{time.perf_counter() - start:.0f}s]",
                flush=True,
            )
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            # Keep the copy in CPU memory, so it does not count towards GPU use.
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    seconds = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    model.load_state_dict(best_state)
    pred, _ = predict(model, val_loader, loss_fn, device)
    n_params = sum(p.numel() for p in model.parameters())
    del model, optimizer, scaler
    torch.cuda.empty_cache()  # free GPU memory before the next seed
    return {
        "seed": seed, "metrics": common.evaluate(val_labels, pred), "pred": pred,
        "history": history, "best_epoch": best_epoch, "epochs_run": len(history),
        "seconds": seconds, "peak_gpu_gb": peak_gb, "state": best_state,
        "n_params": n_params, "config": asdict(cfg),
    }
