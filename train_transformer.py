"""Step 4: fine-tune a pretrained transformer over 3 seeds.

Run:  .\\.venv\\Scripts\\python.exe train_transformer.py
      .\\.venv\\Scripts\\python.exe train_transformer.py --model roberta

The data split never changes; the seeds only vary the head's initial weights,
batch order and dropout. test.json is never opened here.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

import common
import transformer

SEEDS = [42, 43, 44]
MODELS = {
    "distilbert": ("distilbert-base-uncased", "DistilBERT", {}),
    # 3 epochs matched 4 epochs within seed noise on rerun, and trains about 25% faster.
    "roberta": ("roberta-base", "RoBERTa", {"epochs": 3}),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="distilbert")
    key = parser.parse_args().model
    model_name, label, tuned = MODELS[key]
    hf_logging.set_verbosity_error()  # hides the expected "unused weights" load report

    cfg = transformer.Config(model_name=model_name, **tuned)
    clean, _ = common.clean_train(common.load_raw_train())
    train_df, val_df = common.make_split(clean)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lengths = np.array([len(ids) for ids in tokenizer(list(train_df["reviews"]))["input_ids"]])
    print(f"{label} tokens per training review: median {np.median(lengths):.0f}, "
          f"95th percentile {np.percentile(lengths, 95):.0f}, max {lengths.max()}")
    for limit in (128, 192, 256, 512):
        print(f"  cut at {limit}: {(lengths > limit).mean():.1%} of reviews")
    print(flush=True)

    runs = [transformer.train_once(train_df, val_df, cfg, seed) for seed in SEEDS]

    per_seed = pd.DataFrame([{
        "seed": r["seed"], **{f"val_{k}": v for k, v in r["metrics"].items()},
        "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
        "train_seconds": r["seconds"], "peak_gpu_gb": r["peak_gpu_gb"], "n_params": r["n_params"],
    } for r in runs])
    common.RESULTS.mkdir(exist_ok=True)
    per_seed.to_csv(common.RESULTS / f"{key}_runs.csv", index=False)
    history = pd.DataFrame([row for r in runs for row in r["history"]])
    history.to_csv(common.RESULTS / f"{key}_history.csv", index=False)

    # Report the mean; use the best seed's model only for later predictions.
    best = max(runs, key=lambda r: r["metrics"]["macro_f1"])
    columns = ["val_accuracy", "val_macro_f1", "val_neg_f1", "val_neg_precision", "val_neg_recall"]
    common.update_comparison({
        "model": f"{label} (fine-tuned)",
        **{c: per_seed[c].mean() for c in columns},
        **{f"{c}_std": per_seed[c].std(ddof=1) for c in columns},
        "runs": len(runs), "train_seconds": per_seed["train_seconds"].mean(),
        "peak_gpu_gb": per_seed["peak_gpu_gb"].max(), "n_params": best["n_params"],
    })

    y_val = val_df["sentiments"].to_numpy()
    common.save_val_predictions(val_df, best["pred"], key)
    common.save_confusion_matrix(y_val, best["pred"], f"{label} (seed {best['seed']})", key)
    common.save_learning_curves(history, runs, best["seed"], label, key)

    models = common.ROOT / "models"
    models.mkdir(exist_ok=True)
    torch.save({"state": best["state"], "config": best["config"], "seed": best["seed"]},
               models / f"{key}.pt")

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
