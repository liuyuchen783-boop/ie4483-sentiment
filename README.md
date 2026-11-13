# IE4483 Mini Project: Sentiment Classification of Product Reviews

Classifies product reviews as positive (1) or negative (0).

Not included in this repository: the course data (`train.json`, `test.json`) and the trained
models in `models/`. Put the two data files in this folder before running anything; the training
scripts below recreate the models. Output files that quote full reviews (`submission.csv`,
`error_analysis.csv`, `results/test_predictions.csv` and `results/val_predictions/`) are also left
out; the training scripts, `make_submission.py` and `error_analysis.py` recreate them.

Claude Code, an AI assistant from Anthropic, was used to write the code, run the experiments and
draft the report.

## Setup (Windows, PowerShell, from this folder)

Tested with Python 3.14.2 and an NVIDIA RTX 4070 Laptop GPU (driver 555.97).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Call `.\.venv\Scripts\python.exe` directly rather than activating the environment,
because the `&` in this folder's path can break command-prompt scripts.

## Run

| Step | Command | Output |
|---|---|---|
| Check the GPU and environment | `.\.venv\Scripts\python.exe check_env.py` | Printed pass or fail |
| TF-IDF baseline | `.\.venv\Scripts\python.exe train_baseline.py` | `results/` (see below) |
| BiLSTM, 3 seeds (about 3 minutes) | `.\.venv\Scripts\python.exe train_bilstm.py` | `results/`, `models/bilstm.pt` |
| DistilBERT, 3 seeds (about 8 minutes) | `.\.venv\Scripts\python.exe train_transformer.py` | `results/`, `models/distilbert.pt` |
| RoBERTa, 3 seeds (about 13 minutes) | `.\.venv\Scripts\python.exe train_transformer.py --model roberta` | `results/`, `models/roberta.pt` |
| Tuning sweep (minutes to about 1.5 hours each) | `.\.venv\Scripts\python.exe tune.py baselines` (or `bilstm`, `distilbert`, `roberta`) | `results/tuning/` |
| Tuning summaries and plots | `.\.venv\Scripts\python.exe plot_tuning.py` | `results/tuning/` |
| Test predictions (needs `models/roberta.pt`) | `.\.venv\Scripts\python.exe make_submission.py` | `submission.csv` |
| Error analysis (needs all files in `models/`) | `.\.venv\Scripts\python.exe error_analysis.py` | `error_analysis.csv`, `results/` |
| Try your own sentence (needs `models/roberta.pt`) | `.\.venv\Scripts\python.exe predict.py` | Printed label and confidence |
| Web page for trying the model (first run `.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt`) | `.\.venv\Scripts\python.exe demo.py` | Password-protected page at `http://127.0.0.1:7860` |

To host the demo without this computer, sign in with `.\.venv\Scripts\hf.exe auth login`, then run `.\.venv\Scripts\python.exe deploy_space.py <username>/review-sentiment-tester`. This uploads `hf_space/` and `models/roberta.pt` to a Hugging Face Space, which runs on CPU.

`demo.py --share` asks Gradio for a public link. On the development laptop, Windows application control blocks Gradio's tunnel program (WinError 4557), so no link is created there.

The scripts already use the final tuned settings, so the sweeps only need rerunning to reproduce the tuning evidence. The saved checkpoints in `models/` are not needed to rerun training. Retraining RoBERTa overwrites `models/roberta.pt` and can give slightly different weights, because GPU runs here are not bit-exact.

## Files

- `common.py`: data cleaning, the shared 85/15 split, metrics and plotting
- `check_env.py`: trains on one batch to confirm GPU and fp16 work (not model results)
- `train_baseline.py`: TF-IDF with Linear SVM and Logistic Regression
- `bilstm.py`: the from-scratch tokenizer, vocabulary, model and training loop
- `train_bilstm.py`: trains the BiLSTM over 3 seeds and saves its results
- `transformer.py`: the fine-tuned transformer model and training loop
- `train_transformer.py`: trains DistilBERT or RoBERTa over 3 seeds and saves results
- `tune.py`: changes one setting at a time from the default, over 3 seeds
- `plot_tuning.py`: summarises and plots the tuning sweeps
- `make_submission.py`: predicts `test.json` with the saved RoBERTa model and checks the file
- `error_analysis.py`: collects every model's mistakes and the hand-picked hard test reviews
- `notes.md`: summary of the results and reasoning for report parts (a) to (f)

## Results

- `results/comparison.csv`: all models on the shared validation split
- `results/baseline_cv.csv`: 5-fold cross-validation, with and without repeated reviews
- `results/logreg_top_terms.csv`: the terms Logistic Regression weights most heavily
- `results/confusion_matrices/`: one plot per model
- `results/val_predictions/`: each model's validation predictions
- `results/{bilstm,distilbert,roberta}_runs.csv` and `_history.csv`: per-seed and per-epoch results
- `results/learning_curves/`: neural model training plots
- `results/tuning/`: tuning runs, summaries and plots
- `submission.csv`: test predictions from the selected model, RoBERTa (seed 44)
- `error_analysis.csv`: misclassified validation reviews and hard test reviews, with each model's prediction. Test labels here are one person's judgement, not ground truth
- `results/test_predictions.csv`: each model's prediction for every test review
- `results/error_rates_by_pattern.csv`: validation error rates for reviews with negation or contrast words

## Reproducibility

The validation split uses a fixed seed (42), so every model is scored on the same
1,078 reviews. Repeated reviews are removed from `train.json` before splitting.
`test.json` is not used for training or tuning. Only `make_submission.py` and
`error_analysis.py` read it, and only to make predictions.
