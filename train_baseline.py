"""Step 2: TF-IDF baseline with Linear SVM and Logistic Regression.

Run:  .\\.venv\\Scripts\\python.exe train_baseline.py

Part 1 is 5-fold cross-validation, to compare with the earlier 0.93 / 0.85 and to
measure how much repeated reviews inflate scores.
Part 2 is the fixed 85/15 split shared with the neural models.
test.json is never opened here.
"""
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import common

SEED = 42


def build(classifier):
    # A Pipeline refits the vectorizer on each training fold only, so validation
    # words can never influence the features (no data leakage).
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # word pairs capture negation, e.g. "not good"
        sublinear_tf=True,  # 1 + log(tf): the 5th "great" adds less than the 2nd
        min_df=2,  # drop terms seen in one review only: mostly typos and noise
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def classifiers(svm_c=1.0, logreg_c=1.0):
    # class_weight='balanced' makes each negative review count ~3.4x and each
    # positive ~0.6x. C defaults to 1 for the cross-validation check.
    return {
        "svm": ("TF-IDF + Linear SVM", LinearSVC(C=svm_c, class_weight="balanced", random_state=SEED)),
        "logreg": (
            "TF-IDF + Logistic Regression",
            LogisticRegression(C=logreg_c, class_weight="balanced", max_iter=5000, random_state=SEED),
        ),
    }


def cross_validate(df, data_label):
    rows = []
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    x, y = df["reviews"].to_numpy(), df["sentiments"].to_numpy()
    for key, (name, classifier) in classifiers().items():
        scores = []
        for train_idx, val_idx in folds.split(x, y):
            model = build(classifier).fit(x[train_idx], y[train_idx])
            scores.append(common.evaluate(y[val_idx], model.predict(x[val_idx])))
        frame = pd.DataFrame(scores)
        rows.append({
            "data": data_label, "rows": len(df), "model": name,
            **{f"{m}_mean": frame[m].mean() for m in ("accuracy", "macro_f1", "neg_f1")},
            **{f"{m}_std": frame[m].std(ddof=1) for m in ("accuracy", "macro_f1", "neg_f1")},
        })
    return rows


def main():
    common.RESULTS.mkdir(exist_ok=True)
    raw = common.load_raw_train()
    clean, stats = common.clean_train(raw)
    print("Data cleaning:", stats)
    train_df, val_df = common.make_split(clean)
    print(
        f"Split: {len(train_df)} train ({(train_df.sentiments == 0).mean():.1%} negative), "
        f"{len(val_df)} validation ({(val_df.sentiments == 0).mean():.1%} negative)\n"
    )

    cv = pd.DataFrame(cross_validate(raw, "all rows") + cross_validate(clean, "repeats removed"))
    cv.to_csv(common.RESULTS / "baseline_cv.csv", index=False)
    print("5-fold cross-validation (mean +/- std):")
    for row in cv.itertuples():
        print(
            f"  {row.data:<16} {row.model:<29} acc {row.accuracy_mean:.4f}+/-{row.accuracy_std:.4f}  "
            f"macro-F1 {row.macro_f1_mean:.4f}+/-{row.macro_f1_std:.4f}  "
            f"neg-F1 {row.neg_f1_mean:.4f}+/-{row.neg_f1_std:.4f}"
        )

    x_train, y_train = train_df["reviews"], train_df["sentiments"]
    x_val, y_val = val_df["reviews"], val_df["sentiments"]

    # Reference point: what you score by learning nothing at all.
    dummy = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    dummy_metrics = common.evaluate(y_val, dummy.predict(x_val))
    common.update_comparison({
        "model": "Always predict positive", **{f"val_{k}": v for k, v in dummy_metrics.items()},
        "runs": 1, "train_seconds": 0.0, "peak_gpu_gb": 0.0, "n_params": 0,
    })

    print("\nFixed 85/15 validation split:")
    print(f"  {'Always predict positive':<29} acc {dummy_metrics['accuracy']:.4f}  "
          f"macro-F1 {dummy_metrics['macro_f1']:.4f}  neg-F1 {dummy_metrics['neg_f1']:.4f}")

    fitted = {}
    # Best C from the step 5 sweep (results/tuning/baselines.csv).
    for key, (name, classifier) in classifiers(svm_c=3.0, logreg_c=10.0).items():
        start = time.perf_counter()
        model = build(classifier).fit(x_train, y_train)
        seconds = time.perf_counter() - start
        fitted[key] = model
        pred = model.predict(x_val)
        metrics = common.evaluate(y_val, pred)
        n_features = len(model["tfidf"].vocabulary_)
        table = common.update_comparison({
            "model": name, **{f"val_{k}": v for k, v in metrics.items()},
            "runs": 1, "train_seconds": seconds, "peak_gpu_gb": 0.0,
            "n_params": n_features + 1,  # one weight per feature plus the intercept
        })
        common.save_val_predictions(val_df, pred, key)
        common.save_confusion_matrix(y_val, pred, name, key)
        print(f"  {name:<29} acc {metrics['accuracy']:.4f}  macro-F1 {metrics['macro_f1']:.4f}  "
              f"neg-F1 {metrics['neg_f1']:.4f}  (neg precision {metrics['neg_precision']:.3f}, "
              f"recall {metrics['neg_recall']:.3f})  {seconds:.2f}s, {n_features:,} features")

    # The largest weights show which terms the model relies on: useful for the report.
    model = fitted["logreg"]
    terms = np.array(model["tfidf"].get_feature_names_out())
    weights = model["clf"].coef_[0]
    order = np.argsort(weights)
    top = pd.DataFrame({
        "most_negative_term": terms[order[:25]], "negative_weight": weights[order[:25]],
        "most_positive_term": terms[order[::-1][:25]], "positive_weight": weights[order[::-1][:25]],
    })
    top.to_csv(common.RESULTS / "logreg_top_terms.csv", index=False)
    print("\nStrongest negative terms:", ", ".join(top["most_negative_term"].head(15)))
    print("Strongest positive terms:", ", ".join(top["most_positive_term"].head(15)))
    negation = [t for t in terms[order[:300]] if t.split()[0] in {"not", "no", "don", "never", "isn", "wasn", "didn"}]
    print("Negation pairs among the 300 most negative terms:", ", ".join(negation[:15]))
    print(f"\nSaved results to {common.RESULTS}")


if __name__ == "__main__":
    main()
