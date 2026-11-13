"""Step 7: collect every model's mistakes, for report question (e).

Run:  .\\.venv\\Scripts\\python.exe error_analysis.py

Writes error_analysis.csv and results/test_predictions.csv. For each neural model it
uses the best seed's saved checkpoint. SVM and Logistic Regression were not saved,
so they are refitted on the same training data. Nothing is tuned here, and no model
sees test.json except to predict it.

Every model is first checked against the validation predictions saved after training,
so the table cannot silently describe different models from the ones reported.
"""
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

import bilstm
import common
import train_baseline
import transformer

MODELS = ["svm", "logreg", "bilstm", "distilbert", "roberta"]
OUTPUT = common.ROOT / "error_analysis.csv"
DEVICE = torch.device("cuda")

# Test reviews chosen by reading them. test.json has no labels, so "label" is my own
# judgement of the writer's overall sentiment, and is not ground truth.
# {row in test.json: (category, label, why it is hard)}
TEST_EXAMPLES = {
    1782: ("negation", 0, "'not too crazy about' is a polite negative with no strongly negative word"),
    1612: ("negation", 0, "'not the best' negates a positive word"),
    551: ("negation", 1, "'not disappointed' negates a negative word"),
    334: ("negation", 1, "'not too short, but not too long' praises the fit using negated words"),
    1217: ("negation", 1, "'I don't think this is an ugly bra' negates a negative word"),
    159: ("sarcasm", 0, "the bra is rejected and joked about as a slingshot, in cheerful wording"),
    1720: ("sarcasm", 0, "'What I love most' is ironic: only the upper halves of the fingers are cold"),
    1827: ("sarcasm", 0, "grants one strength, then mocks it: 'I could get the same results with duct tape'"),
    80: ("sarcasm", 0, "'should really be called Simulation Leather' mocks the material"),
    930: ("sarcasm", 0, "'I can't imagine who could pull off these clown pants' is a rhetorical put-down"),
    549: ("mixed", 0, "'I love these boots' but the strap broke and the writer wants a return"),
    754: ("mixed", 0, "'well made' and 'nice' but too small to be useful"),
    99: ("mixed", 0, "liked at first, then the underwire broke through"),
    1227: ("mixed", 0, "'cute and comfortable', but not leather as described and not worth the price paid"),
    302: ("mixed", 1, "'not the best' quality and it snags, but 'it's great nonetheless'"),
}

# RoBERTa's validation errors, by row in train.json, each given one main cause by reading it.
# These tags are my judgement. A review can show several causes, and only one is recorded.
ROBERTA_ERROR_TAGS = {
    **dict.fromkeys(
        [1378, 2185, 2476, 2822, 3196, 3212, 3976, 4542, 5551, 6637, 6722,
         467, 1503, 1616, 2023, 2473, 3478, 4310, 4325, 4810],
        "mixed sentiment"),
    **dict.fromkeys([2258, 5348, 5349, 3604], "sarcasm or humour"),
    **dict.fromkeys([3367, 6902, 7234, 33, 1132, 4632], "implied opinion, few sentiment words"),
    **dict.fromkeys([204, 550, 954, 2144], "label looks doubtful"),
    **dict.fromkeys([4801, 2531, 3289, 4873], "no clear opinion"),
}
NEGATION = r"\b(?:not|no|never|none|nothing|without|cannot)\b|n't\b"
CONTRAST = r"\b(?:but|however|although|though|except|otherwise|while)\b"


def predict_baselines(train_df, texts_by_split):
    out = {}
    for key, (_, classifier) in train_baseline.classifiers(svm_c=3.0, logreg_c=10.0).items():
        model = train_baseline.build(classifier).fit(train_df["reviews"], train_df["sentiments"])
        out[key] = {split: model.predict(texts).astype(int) for split, texts in texts_by_split.items()}
    return out


def predict_bilstm(texts_by_split):
    saved = torch.load(common.ROOT / "models" / "bilstm.pt", map_location="cpu")
    cfg = bilstm.Config(**saved["config"])
    model = bilstm.BiLSTMClassifier(len(saved["vocab"]), cfg)
    model.load_state_dict(saved["state"])
    model.to(DEVICE).eval()
    out = {}
    for split, texts in texts_by_split.items():
        data = bilstm.ReviewDataset(texts, [0] * len(texts), saved["vocab"], cfg.max_len)
        loader = DataLoader(data, batch_size=128, collate_fn=bilstm.collate)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            out[split] = torch.cat([model(ids.to(DEVICE)).argmax(dim=1).cpu() for ids, _ in loader]).numpy()
    return out


def predict_transformer(key, texts_by_split):
    """Returns predictions and the probability of positive, which shows confidence."""
    saved = torch.load(common.ROOT / "models" / f"{key}.pt", map_location="cpu")
    cfg = transformer.Config(**saved["config"])
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = transformer.TransformerClassifier(cfg.model_name, cfg.dropout)
    model.load_state_dict(saved["state"])
    model.to(DEVICE).eval()
    pred, prob = {}, {}
    for split, texts in texts_by_split.items():
        loader = transformer.make_loader(texts, [0] * len(texts), tokenizer, cfg, batch_size=64)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            logits = torch.cat([model(ids.to(DEVICE), mask.to(DEVICE)).float().cpu() for ids, mask, _ in loader])
        pred[split] = logits.argmax(dim=1).numpy()
        prob[split] = torch.softmax(logits, dim=1)[:, 1].numpy()
    del model
    torch.cuda.empty_cache()
    return pred, prob


def main():
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    clean, _ = common.clean_train(common.load_raw_train())
    train_df, val_df = common.make_split(clean)
    with open(common.ROOT / "test.json", encoding="utf-8") as file:
        test_reviews = [row["reviews"] for row in json.load(file)]
    texts = {"val": list(val_df["reviews"]), "test": test_reviews}

    preds = predict_baselines(train_df, texts)
    preds["bilstm"] = predict_bilstm(texts)
    preds["distilbert"], _ = predict_transformer("distilbert", texts)
    preds["roberta"], prob = predict_transformer("roberta", texts)

    # Check: every model must reproduce the validation predictions saved after training.
    y_val = val_df["sentiments"].to_numpy()
    print(f"{'Model':<12}{'Differ from saved':>18}{'Wrong':>7}{'Macro-F1':>10}")
    for key in MODELS:
        saved = pd.read_csv(common.RESULTS / "val_predictions" / f"{key}.csv")
        assert saved["train_index"].tolist() == val_df["train_index"].tolist()
        differ = int((saved["prediction"].to_numpy() != preds[key]["val"]).sum())
        assert differ == 0, f"{key}: {differ} predictions differ from the saved ones"
        wrong = int((preds[key]["val"] != y_val).sum())
        print(f"{key:<12}{differ:>18}{wrong:>7}{common.evaluate(y_val, preds[key]['val'])['macro_f1']:>10.4f}")

    test_table = pd.DataFrame({"test_row": range(len(test_reviews)), "reviews": test_reviews,
                               **{key: preds[key]["test"] for key in MODELS},
                               "roberta_prob_positive": prob["test"].round(4)})
    test_table.to_csv(common.RESULTS / "test_predictions.csv", index=False)

    # Validation: every review at least one model got wrong.
    val_table = pd.DataFrame({
        "source": "validation", "category": "misclassified", "row": val_df["train_index"].to_numpy(),
        "reviews": val_df["reviews"].to_numpy(), "label": y_val, "label_source": "train.json",
        **{key: preds[key]["val"] for key in MODELS}, "roberta_prob_positive": prob["val"].round(4),
    })
    val_table["models_wrong"] = sum((val_table[key] != val_table["label"]).astype(int) for key in MODELS)
    val_table["note"] = val_table["row"].map(ROBERTA_ERROR_TAGS).fillna("")
    misclassified = val_table[val_table["models_wrong"] > 0].sort_values(
        ["models_wrong", "row"], ascending=[False, True])

    # The tags must cover exactly RoBERTa's errors, so the counts in notes.md are complete.
    roberta_wrong = val_table[val_table["roberta"] != val_table["label"]]
    assert set(roberta_wrong["row"]) == set(ROBERTA_ERROR_TAGS)
    confidence = (roberta_wrong["roberta_prob_positive"] - 0.5).abs() + 0.5
    print("\nRoBERTa errors by main cause (my reading):")
    print(roberta_wrong.groupby(["note", "label"]).size().unstack(fill_value=0).to_string())
    print(f"RoBERTa errors made with over 90% confidence: {(confidence > 0.9).sum()} of {len(roberta_wrong)}")

    # Word patterns are only a rough guide to what makes a review hard, so these are
    # error rates for reviews containing the words, not proof that they caused the errors.
    has_negation = val_table["reviews"].str.contains(NEGATION, case=False, regex=True)
    has_contrast = val_table["reviews"].str.contains(CONTRAST, case=False, regex=True)
    groups = {"negation word": has_negation, "contrast word": has_contrast,
              "both": has_negation & has_contrast, "neither": ~has_negation & ~has_contrast,
              "all": pd.Series(True, index=val_table.index)}
    rates = pd.DataFrame([{
        "reviews_containing": name, "reviews": int(mask.sum()),
        "negative_share": (val_table.loc[mask, "label"] == 0).mean(),
        **{f"{key}_error_rate": (val_table.loc[mask, key] != val_table.loc[mask, "label"]).mean() for key in MODELS},
    } for name, mask in groups.items()])
    rates.to_csv(common.RESULTS / "error_rates_by_pattern.csv", index=False)
    print("\nValidation error rate by words in the review:")
    print(rates.round(3).to_string(index=False))

    # Test: the hand-picked hard examples.
    rows = []
    for row, (category, label, note) in TEST_EXAMPLES.items():
        item = test_table.iloc[row]
        rows.append({"source": "test", "category": category, "row": row, "reviews": item["reviews"],
                     "label": label, "label_source": "my judgement, not ground truth",
                     **{key: int(item[key]) for key in MODELS},
                     "roberta_prob_positive": item["roberta_prob_positive"],
                     "models_wrong": sum(int(item[key]) != label for key in MODELS), "note": note})
    table = pd.concat([misclassified, pd.DataFrame(rows, columns=val_table.columns)], ignore_index=True)
    table.to_csv(OUTPUT, index=False)

    print(f"\nWrote {OUTPUT.name}: {len(misclassified)} validation reviews, {len(rows)} test reviews")
    print("Validation reviews by number of models wrong:")
    print(misclassified["models_wrong"].value_counts().sort_index().to_string())
    for name, group in misclassified.groupby("label"):
        print(f"True label {name}: {len(group)} reviews with at least one model wrong")
    print("\nHand-picked test reviews, label is my judgement:")
    picks = table[table["source"] == "test"]
    print(picks[["category", "row", "label", *MODELS, "roberta_prob_positive", "models_wrong"]].to_string(index=False))
    print(picks.groupby("category")[MODELS].apply(
        lambda group: pd.Series({key: int((group[key] != picks.loc[group.index, "label"]).sum()) for key in MODELS})
    ).to_string())
    print("\nAgreement between models on test.json (share of rows with the same prediction):")
    print(pd.DataFrame({a: {b: np.mean(preds[a]["test"] == preds[b]["test"]) for b in MODELS} for a in MODELS}).round(3))


if __name__ == "__main__":
    main()
