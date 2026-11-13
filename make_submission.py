"""Step 6: predict test.json with the selected model and write submission.csv.

Run:  .\\.venv\\Scripts\\python.exe make_submission.py

The model was chosen on validation macro-F1 before test.json was opened. Nothing
is fitted here: test.json only passes through the saved tokenizer and model.
Test reviews are not deduplicated or reordered, because every row needs a prediction.
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

import common
import transformer

CHECKPOINT = common.ROOT / "models" / "roberta.pt"
TEST_PATH = common.ROOT / "test.json"
OUTPUT = common.ROOT / "submission.csv"
EXPECTED_ROWS = 1851


def predict_texts(model, texts, tokenizer, cfg, device):
    # predict() also computes a loss, so it needs labels. Test reviews have none:
    # these zeros are placeholders, and that loss is thrown away.
    loader = transformer.make_loader(texts, [0] * len(texts), tokenizer, cfg, batch_size=64)
    pred, _ = transformer.predict(model, loader, nn.CrossEntropyLoss(), device)
    return pred


def main():
    hf_logging.set_verbosity_error()
    device = torch.device("cuda")
    saved = torch.load(CHECKPOINT, map_location="cpu")
    cfg = transformer.Config(**saved["config"])
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = transformer.TransformerClassifier(cfg.model_name, cfg.dropout)
    model.load_state_dict(saved["state"])
    model.to(device)
    print(f"Loaded {cfg.model_name}, seed {saved['seed']}, {cfg.epochs} epochs, max length {cfg.max_len}")

    # Check 1: the loaded model must reproduce the predictions saved after training.
    clean, _ = common.clean_train(common.load_raw_train())
    _, val_df = common.make_split(clean)
    val_pred = predict_texts(model, val_df["reviews"], tokenizer, cfg, device)
    metrics = common.evaluate(val_df["sentiments"].to_numpy(), val_pred)
    saved_pred = pd.read_csv(common.RESULTS / "val_predictions" / "roberta.csv")["prediction"].to_numpy()
    changed = int((val_pred != saved_pred).sum())
    print(f"Validation: accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f}, "
          f"neg-F1 {metrics['neg_f1']:.4f}; {changed} of {len(val_pred)} differ from the saved predictions")
    assert changed == 0, "The checkpoint does not match the saved validation predictions"

    with open(TEST_PATH, encoding="utf-8") as file:
        reviews = [row["reviews"] for row in json.load(file)]
    pred = predict_texts(model, reviews, tokenizer, cfg, device)
    submission = pd.DataFrame({"reviews": reviews, "predictions": pred.astype(int)})

    # Check 2: the file format, tested before the placeholder is replaced.
    assert list(submission.columns) == ["reviews", "predictions"]
    assert len(submission) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(submission)}"
    assert set(np.unique(pred)) <= {0, 1}
    assert submission["reviews"].tolist() == reviews
    submission.to_csv(OUTPUT, index=False)

    # Check 3: read the written file back, as a marker would.
    written = pd.read_csv(OUTPUT, keep_default_na=False)
    same_text = sum(" ".join(a.split()) == " ".join(b.split()) for a, b in zip(written["reviews"], reviews))
    assert written.shape == (EXPECTED_ROWS, 2) and same_text == EXPECTED_ROWS
    assert set(written["predictions"].unique()) <= {0, 1}

    counts = submission["predictions"].value_counts().sort_index()
    print(f"Wrote {OUTPUT.name}: {len(written)} rows; "
          + ", ".join(f"{label}: {count} ({count / len(pred):.1%})" for label, count in counts.items()))
    print(f"Training data for comparison: {(clean['sentiments'] == 0).mean():.1%} negative")


if __name__ == "__main__":
    main()
