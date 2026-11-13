"""Step 3: train the from-scratch BiLSTM over 3 seeds.

Run:  .\\.venv\\Scripts\\python.exe train_bilstm.py

The data split never changes; the seeds only vary weight initialisation, batch
order and dropout. So the spread shows how much a result depends on random luck.
test.json is never opened here.
"""
import numpy as np
import pandas as pd
import torch

import bilstm
import common

SEEDS = [42, 43, 44]


def main():
    cfg = bilstm.Config()
    clean, _ = common.clean_train(common.load_raw_train())
    train_df, val_df = common.make_split(clean)

    vocab = bilstm.build_vocab(train_df["reviews"], cfg.min_freq)
    all_words = {t for text in train_df["reviews"] for t in bilstm.tokenize(text)}
    lengths = np.array([len(bilstm.tokenize(t)) for t in train_df["reviews"]])
    val_tokens = [t for text in val_df["reviews"] for t in bilstm.tokenize(text)]
    print(f"Vocabulary: {len(vocab):,} entries, from {len(all_words):,} distinct training words")
    print(f"Review tokens: median {np.median(lengths):.0f}, 95th percentile {np.percentile(lengths, 95):.0f}, "
          f"max {lengths.max()}; {(lengths > cfg.max_len).mean():.1%} cut at {cfg.max_len}")
    print(f"Validation tokens outside the vocabulary (<unk>): "
          f"{np.mean([t not in vocab for t in val_tokens]):.1%}\n")

    runs = [bilstm.train_once(train_df, val_df, cfg, seed) for seed in SEEDS]

    per_seed = pd.DataFrame([{
        "seed": r["seed"], **{f"val_{k}": v for k, v in r["metrics"].items()},
        "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
        "train_seconds": r["seconds"], "peak_gpu_gb": r["peak_gpu_gb"], "n_params": r["n_params"],
    } for r in runs])
    common.RESULTS.mkdir(exist_ok=True)
    per_seed.to_csv(common.RESULTS / "bilstm_runs.csv", index=False)
    history = pd.DataFrame([row for r in runs for row in r["history"]])
    history.to_csv(common.RESULTS / "bilstm_history.csv", index=False)

    # Report the mean; use the best seed's model only for later predictions.
    best = max(runs, key=lambda r: r["metrics"]["macro_f1"])
    columns = ["val_accuracy", "val_macro_f1", "val_neg_f1", "val_neg_precision", "val_neg_recall"]
    common.update_comparison({
        "model": "BiLSTM (from scratch)",
        **{c: per_seed[c].mean() for c in columns},
        **{f"{c}_std": per_seed[c].std(ddof=1) for c in columns},
        "runs": len(runs), "train_seconds": per_seed["train_seconds"].mean(),
        "peak_gpu_gb": per_seed["peak_gpu_gb"].max(), "n_params": best["n_params"],
    })

    y_val = val_df["sentiments"].to_numpy()
    common.save_val_predictions(val_df, best["pred"], "bilstm")
    common.save_confusion_matrix(y_val, best["pred"], f"BiLSTM (seed {best['seed']})", "bilstm")
    common.save_learning_curves(history, runs, best["seed"], "BiLSTM", "bilstm")

    models = common.ROOT / "models"
    models.mkdir(exist_ok=True)
    torch.save({"state": best["state"], "vocab": best["vocab"], "config": best["config"],
                "seed": best["seed"]}, models / "bilstm.pt")

    print("\nPer seed:")
    print(per_seed.round(4).to_string(index=False))
    print("\nMean +/- std over 3 seeds:")
    for c in columns:
        print(f"  {c:<18} {per_seed[c].mean():.4f} +/- {per_seed[c].std(ddof=1):.4f}")
    print(f"  train seconds      {per_seed['train_seconds'].mean():.1f}")
    print(f"  peak GPU           {per_seed['peak_gpu_gb'].max():.2f} GB")
    print(f"  parameters         {best['n_params']:,}")
    print(f"Best seed: {best['seed']}")


if __name__ == "__main__":
    main()
