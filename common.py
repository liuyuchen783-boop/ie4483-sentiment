"""Shared helpers, so every model sees the same data and is scored the same way."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write plots to files; no display window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SPLIT_SEED = 42  # never varies: all models and all seeds share one validation set
VAL_FRACTION = 0.15
# Validated categorical chart colours, always assigned in this order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def load_raw_train():
    with open(ROOT / "train.json", encoding="utf-8") as handle:
        return pd.DataFrame(json.load(handle))


def clean_train(raw):
    """Remove repeated reviews so one review cannot be in both train and validation.

    Reviews count as repeats if they match after lowercasing and collapsing
    whitespace. If the copies disagree on the label, all of them are dropped,
    because we cannot tell which label is right. Only training data goes through
    this: test.json must keep every row for submission.csv.
    """
    df = raw.copy()
    df["train_index"] = df.index  # position in train.json, for tracing errors back
    key = df["reviews"].map(lambda text: " ".join(text.lower().split()))
    conflicting = df.groupby(key)["sentiments"].transform("nunique") > 1
    kept = df[~conflicting][~key[~conflicting].duplicated()].reset_index(drop=True)
    stats = {
        "raw_rows": len(df),
        "conflicting_rows_dropped": int(conflicting.sum()),
        "repeat_rows_dropped": len(df) - int(conflicting.sum()) - len(kept),
        "clean_rows": len(kept),
    }
    return kept, stats


def make_split(clean):
    """The fixed stratified 85/15 split. Stratifying keeps ~15% negatives in both parts."""
    return train_test_split(
        clean, test_size=VAL_FRACTION, stratify=clean["sentiments"], random_state=SPLIT_SEED
    )


def evaluate(y_true, y_pred):
    """Accuracy alone flatters imbalanced data, so also report the negative class."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "neg_f1": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "neg_precision": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "neg_recall": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
    }


def update_comparison(row):
    """Add or replace one model's row in results/comparison.csv."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "comparison.csv"
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not table.empty:
        table = table[table["model"] != row["model"]]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(path, index=False)
    return table


def save_val_predictions(val_df, y_pred, model_key):
    """Keep each model's validation predictions; the error analysis merges these."""
    folder = RESULTS / "val_predictions"
    folder.mkdir(parents=True, exist_ok=True)
    out = val_df[["train_index", "reviews", "sentiments"]].copy()
    out["prediction"] = y_pred
    out.to_csv(folder / f"{model_key}.csv", index=False)


def _style_line_chart(ax, ylabel):
    ink = "#52514e"
    ax.set_xlabel("Epoch", color=ink)
    ax.set_ylabel(ylabel, color=ink)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=ink, length=0)
    ax.xaxis.get_major_locator().set_params(integer=True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")


def save_learning_curves(history, runs, best_seed, label, model_key):
    """Macro-F1 per seed, plus train vs validation loss for the best seed.

    runs is a list of dicts with "seed" and "best_epoch". They are two separate
    figures because the same colours mean different things in each.
    """
    folder = RESULTS / "learning_curves"
    folder.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=200)
    for color, run in zip(COLORS, runs):
        rows = history[history["seed"] == run["seed"]]
        ax.plot(rows["epoch"], rows["val_macro_f1"], color=color, linewidth=2, label=f"Seed {run['seed']}")
        best = rows[rows["epoch"] == run["best_epoch"]]
        ax.scatter(best["epoch"], best["val_macro_f1"], color=color, s=48, zorder=3,
                   edgecolors="white", linewidths=1.5)
    _style_line_chart(ax, "Validation macro-F1")
    ax.set_title(f"{label} validation macro-F1 (dot = epoch kept)", fontsize=11, loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(folder / f"{model_key}_val_macro_f1.png")
    plt.close(fig)

    run = next(r for r in runs if r["seed"] == best_seed)
    rows = history[history["seed"] == best_seed]
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=200)
    ax.plot(rows["epoch"], rows["train_loss"], color=COLORS[0], linewidth=2, label="Training loss")
    ax.plot(rows["epoch"], rows["val_loss"], color=COLORS[1], linewidth=2, label="Validation loss")
    ax.axvline(run["best_epoch"], color="#c9c8c2", linewidth=1, linestyle="--")
    _style_line_chart(ax, "Weighted cross-entropy loss")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{label} loss, seed {best_seed} (dashed line = epoch kept)", fontsize=11, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(folder / f"{model_key}_loss.png")
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, title, model_key):
    """Shade by share of each true class, so the small negative class stays visible."""
    folder = RESULTS / "confusion_matrices"
    folder.mkdir(parents=True, exist_ok=True)
    counts = confusion_matrix(y_true, y_pred, labels=[0, 1])
    shares = counts / counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.4, 3.8), dpi=200)
    image = ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for row in range(2):
        for col in range(2):
            ink = "white" if shares[row, col] > 0.55 else "#1f2933"
            ax.text(
                col, row, f"{counts[row, col]:,}\n{shares[row, col]:.1%}",
                ha="center", va="center", color=ink, fontsize=11,
            )
    ax.set_xticks([0, 1], ["Negative (0)", "Positive (1)"])
    ax.set_yticks([0, 1], ["Negative (0)", "Positive (1)"], rotation=90, va="center")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=11)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Share of true class")
    colorbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(folder / f"{model_key}.png")
    plt.close(fig)
