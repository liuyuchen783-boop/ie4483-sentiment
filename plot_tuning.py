"""Summarise the tuning sweeps and plot them.

Run:  .\\.venv\\Scripts\\python.exe plot_tuning.py

A value is flagged as better than the default only if its mean macro-F1 gain is
larger than the seed-to-seed standard deviation of both. Smaller gains could be luck.
"""
import matplotlib.pyplot as plt
import pandas as pd

import bilstm
import common
import transformer

TUNING = common.RESULTS / "tuning"
DEFAULTS = {
    "bilstm": bilstm.Config(),
    "distilbert": transformer.Config(),
    "roberta": transformer.Config(model_name="roberta-base"),
}
LABELS = {"bilstm": "BiLSTM", "distilbert": "DistilBERT", "roberta": "RoBERTa"}
SETTING_NAMES = {
    "lr": "Learning rate", "dropout": "Dropout", "hidden": "Hidden size", "max_len": "Max length",
    "class_weighting": "Class weighting", "epochs": "Epochs",
}
INK = "#52514e"


def summarise(key):
    path = TUNING / f"{key}.csv"
    if not path.exists():
        return None
    runs = pd.read_csv(path, dtype={"value": str})
    defaults = DEFAULTS[key]
    base = runs[runs["setting"] == "default"]
    rows = []
    for setting in [s for s in runs["setting"].unique() if s != "default"]:
        default_value = getattr(defaults, setting)
        # The default run belongs on every setting's chart, at its default value.
        part = pd.concat([runs[runs["setting"] == setting], base.assign(value=str(default_value))])
        for value, group in part.groupby("value"):
            rows.append({
                "setting": setting, "value": value, "is_default": value == str(default_value),
                "runs": len(group),
                "macro_f1_mean": group["macro_f1"].mean(), "macro_f1_std": group["macro_f1"].std(ddof=1),
                "neg_f1_mean": group["neg_f1"].mean(), "neg_f1_std": group["neg_f1"].std(ddof=1),
                "accuracy_mean": group["accuracy"].mean(), "best_epoch_mean": group["best_epoch"].mean(),
                "peak_gpu_gb": group["peak_gpu_gb"].max(), "n_params": group["n_params"].max(),
            })
    table = pd.DataFrame(rows)
    table["sort"] = pd.to_numeric(table["value"], errors="coerce")
    table = table.sort_values(["setting", "sort", "value"], key=lambda c: c).drop(columns="sort")
    base_f1, base_std = base["macro_f1"].mean(), base["macro_f1"].std(ddof=1)
    table["gain_vs_default"] = table["macro_f1_mean"] - base_f1
    table["better_beyond_noise"] = (
        ~table["is_default"] & (table["gain_vs_default"] > table["macro_f1_std"].clip(lower=base_std))
    )
    table.to_csv(TUNING / f"{key}_summary.csv", index=False)
    return table


def plot_sweep(key, table):
    settings = list(SETTING_NAMES[s] and s for s in SETTING_NAMES if s in set(table["setting"]))
    fig, axes = plt.subplots(1, len(settings), figsize=(2.5 * len(settings) + 0.8, 3.4), dpi=200, sharey=True)
    axes = [axes] if len(settings) == 1 else list(axes)
    for ax, setting in zip(axes, settings):
        part = table[table["setting"] == setting].reset_index(drop=True)
        positions = range(len(part))
        ax.errorbar(positions, part["macro_f1_mean"], yerr=part["macro_f1_std"], color=common.COLORS[0],
                    linewidth=2, marker="o", markersize=7, capsize=3, markeredgecolor="white")
        default = part[part["is_default"]]
        ax.scatter(default.index, default["macro_f1_mean"], s=70, zorder=3, color=common.COLORS[1],
                   edgecolors="white", linewidths=1.2, label="Default")
        ax.set_xticks(list(positions), part["value"], fontsize=8)
        ax.set_xlim(-0.5, len(part) - 0.5)
        ax.set_xlabel(SETTING_NAMES[setting], color=INK)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
    axes[0].set_ylabel("Validation macro-F1", color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(f"{LABELS[key]}: one setting changed at a time (mean and std over 3 seeds)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(TUNING / f"{key}.png")
    plt.close(fig)


def plot_baselines():
    path = TUNING / "baselines.csv"
    if not path.exists():
        return
    runs = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=200)
    for color, (model, label) in zip(common.COLORS, (("svm", "Linear SVM"), ("logreg", "Logistic Regression"))):
        part = runs[runs["model"] == model]
        ax.plot(part["value"], part["macro_f1"], color=color, linewidth=2, marker="o", markersize=6, label=label)
    ax.set_xscale("log")
    ax.set_xticks(sorted(runs["value"].unique()), [f"{v:g}" for v in sorted(runs["value"].unique())])
    ax.minorticks_off()
    ax.set_xlabel("Regularisation setting C (larger = fits the training data harder)", color=INK)
    ax.set_ylabel("Validation macro-F1", color=INK)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.tick_params(colors=INK, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.set_title("TF-IDF baselines: effect of C", fontsize=11, loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(TUNING / "baselines.png")
    plt.close(fig)


def main():
    plot_baselines()
    for key in DEFAULTS:
        table = summarise(key)
        if table is None:
            continue
        plot_sweep(key, table)
        print(f"\n{LABELS[key]}")
        print(table[["setting", "value", "runs", "macro_f1_mean", "macro_f1_std", "neg_f1_mean",
                     "best_epoch_mean", "gain_vs_default", "better_beyond_noise"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
