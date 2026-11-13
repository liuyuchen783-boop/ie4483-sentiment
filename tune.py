"""Step 5: tune one setting at a time, changing it from the default and nothing else.

Run:  .\\.venv\\Scripts\\python.exe tune.py baselines
      .\\.venv\\Scripts\\python.exe tune.py bilstm
      .\\.venv\\Scripts\\python.exe tune.py distilbert
      .\\.venv\\Scripts\\python.exe tune.py roberta

Each model writes its own CSV, so sweeps can run side by side without clashing.
Finished runs are skipped on a restart. Timings here are not for reporting, since
sweeps may share the GPU; reported timings come from the final runs, done alone.
test.json is never opened here.
"""
import argparse
import time
from functools import partial

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from transformers.utils import logging as hf_logging

import bilstm
import common
import train_baseline
import transformer

SEEDS = [42, 43, 44]  # 3 seeds: one seed's noise could be mistaken for a real effect
TUNING = common.RESULTS / "tuning"

SWEEPS = {
    "bilstm": {
        "lr": [3e-4, 1e-3, 3e-3],
        "dropout": [0.2, 0.5, 0.7],
        "hidden": [64, 128, 256],
        "max_len": [100, 200, 400],
        "class_weighting": [True, False],
    },
    "distilbert": {
        "lr": [1e-5, 2e-5, 5e-5],
        "epochs": [2, 3, 4],
        "max_len": [128, 192, 256],
        "dropout": [0.1, 0.3],
        "class_weighting": [True, False],
    },
    # Slower to run, so only the two settings most likely to matter for fine-tuning.
    "roberta": {
        "lr": [1e-5, 2e-5, 3e-5],
        "epochs": [2, 3, 4],
    },
}


def tune_baselines(train_df, val_df):
    """Regularisation strength C: small C means simpler models, large C fits harder."""
    rows = []
    x_train, y_train = train_df["reviews"], train_df["sentiments"]
    x_val, y_val = val_df["reviews"], val_df["sentiments"]
    for c in [0.1, 0.3, 1, 3, 10, 30]:
        for name, classifier in (
            ("svm", LinearSVC(C=c, class_weight="balanced", random_state=42)),
            ("logreg", LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=42)),
        ):
            model = train_baseline.build(classifier).fit(x_train, y_train)
            metrics = common.evaluate(y_val, model.predict(x_val))
            rows.append({"model": name, "setting": "C", "value": c, **metrics})
            print(f"{name} C={c}: macro-F1 {metrics['macro_f1']:.4f}, neg-F1 {metrics['neg_f1']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(TUNING / "baselines.csv", index=False)


def sweep(key, train_df, val_df):
    """Run every (setting, value, seed). The default appears once, shared by all settings."""
    module, cfg_class = {
        "bilstm": (bilstm, bilstm.Config),
        "distilbert": (transformer, transformer.Config),
        "roberta": (transformer, partial(transformer.Config, model_name="roberta-base")),
    }[key]
    defaults = cfg_class()
    path = TUNING / f"{key}.csv"
    if key == "roberta" and not path.exists():
        # The 3-seed RoBERTa run already used the default settings, so reuse it.
        runs = pd.read_csv(common.RESULTS / "roberta_runs.csv")
        runs.columns = [c.removeprefix("val_") for c in runs.columns]
        runs = runs.rename(columns={"train_seconds": "seconds_shared_gpu"})
        runs.insert(0, "value", "default")
        runs.insert(0, "setting", "default")
        # Same column order as the rows appended below.
        runs = runs[[c for c in runs.columns if c != "seconds_shared_gpu"] + ["seconds_shared_gpu"]]
        runs.to_csv(path, index=False)
    done =pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["setting", "value", "seed"])
    finished = {(row.setting, str(row.value), row.seed) for row in done.itertuples()}

    plan = [("default", "default", {})]
    for setting, values in SWEEPS[key].items():
        plan += [(setting, v, {setting: v}) for v in values if v != getattr(defaults, setting)]
    print(f"{key}: {len(plan)} configurations x {len(SEEDS)} seeds = {len(plan) * len(SEEDS)} runs", flush=True)

    for setting, value, change in plan:
        for seed in SEEDS:
            if (setting, str(value), seed) in finished:
                continue
            start = time.perf_counter()
            run = module.train_once(train_df, val_df, cfg_class(**change), seed, verbose=False)
            row = pd.DataFrame([{
                "setting": setting, "value": value, "seed": seed, **run["metrics"],
                "best_epoch": run["best_epoch"], "epochs_run": run["epochs_run"],
                "peak_gpu_gb": run["peak_gpu_gb"], "n_params": run["n_params"],
                "seconds_shared_gpu": time.perf_counter() - start,
            }])
            row.to_csv(path, mode="a", header=not path.exists(), index=False)  # saved after every run
            print(f"{key} {setting}={value} seed {seed}: macro-F1 {run['metrics']['macro_f1']:.4f}, "
                  f"neg-F1 {run['metrics']['neg_f1']:.4f}, {time.perf_counter() - start:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["baselines", "bilstm", "distilbert", "roberta"])
    target = parser.parse_args().target
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    TUNING.mkdir(parents=True, exist_ok=True)

    clean, _ = common.clean_train(common.load_raw_train())
    train_df, val_df = common.make_split(clean)
    if target == "baselines":
        tune_baselines(train_df, val_df)
    else:
        sweep(target, train_df, val_df)


if __name__ == "__main__":
    main()
