# Report notes: parts (a) to (f)

Working notes for writing the report, not report text. All scores are on the same 1,078
validation reviews. Neural models show the mean ± std of 3 seeds (42, 43, 44). Test scores
are unknown, because `test.json` has no labels.

**Headline:** RoBERTa is the selected model, with macro-F1 0.922 ± 0.006, accuracy 0.962 and
negative F1 0.866.

| Model | Accuracy | Macro-F1 | Negative F1 | Train time | Peak GPU | Parameters |
|---|---|---|---|---|---|---|
| Always predict positive | 0.854 | 0.461 | 0.000 | – | – | – |
| BiLSTM | 0.897 ± 0.004 | 0.805 ± 0.008 | 0.671 ± 0.016 | 47 s | 0.08 GB | 1,161,858 |
| TF-IDF + Logistic Regression (C=10) | 0.923 | 0.844 | 0.733 | 0.44 s | None (CPU) | 37,544 |
| TF-IDF + Linear SVM (C=3) | 0.931 | 0.851 | 0.741 | 0.49 s | None (CPU) | 37,544 |
| DistilBERT | 0.947 ± 0.002 | 0.894 ± 0.004 | 0.819 ± 0.007 | 173 s | 1.68 GB | 66,364,418 |
| **RoBERTa** | **0.962 ± 0.003** | **0.922 ± 0.006** | **0.866 ± 0.011** | 263 s | 3.2–5.1 GB | 124,056,578 |

Accuracy alone misleads here: predicting "positive" every time scores 0.854 yet never finds a
negative review. That is why macro-F1 chooses the model.

Source: `results/comparison.csv`.

## (a) Feature format and preprocessing

**Chosen format:** contextual token embeddings from a pretrained language model (RoBERTa). TF-IDF
and learned word embeddings were built for comparison.

**Shared by all models**

1. Load `train.json` (7,401 reviews, 14.6% negative).
2. Remove repeats, matching after lowercasing and collapsing whitespace. This dropped 219 rows
   and left 7,182; no copies disagreed on the label. Why: a review in both training and validation
   inflates scores. In 5-fold cross-validation, the SVM's macro-F1 fell from 0.851 to 0.843 once
   repeats were removed.
3. Split into stratified 85/15 (seed 42): 6,104 training and 1,078 validation reviews, 157 of them
   negative. Why: stratifying keeps the rare class's share equal, and a fixed split means every model
   sees the same reviews.
4. Fit vectorizers and vocabularies on training data only. `test.json` keeps all 1,851 rows, in order.

**By format**

| Step | TF-IDF | BiLSTM | Transformers |
|---|---|---|---|
| Tokens | Words and word pairs (lowercased) | Own regex: words, contractions kept whole, `!` and `?` | The model's own pretrained tokenizer |
| Vocabulary | 37,543 terms, seen in at least 2 reviews | 7,005 entries, including `<pad>` and `<unk>` | Fixed by the pretrained model |
| Rare words | Dropped | Become `<unk>` (2.9% of validation tokens) | Split into known subword pieces |
| Length | No limit | First 200 tokens (cuts 2.1% of reviews) | First 256 tokens (cuts about 2%) |
| Padding | Not needed | To the longest in each batch | To the longest in each batch, with an attention mask |
| Output per review | Sparse vector of 37,543 | Sequence of 128-d embeddings | Sequence of 768-d contextual vectors |

- **Word pairs:** capture negation such as "not good".
- **`sublinear_tf`:** the fifth "great" adds less than the second.
- **No stop-word removal or stemming:** "not" is a stop word and carries the sentiment.
- **Length limits come from the data:** training reviews have a median of 42 BiLSTM tokens (95th percentile
  145) and 53 RoBERTa tokens (95th percentile 180).

## (b) Model, loss and training

**Figure for the selected model.** B = batch size, T = review length in tokens (at most 256).

```
Review text
  │  RoBERTa tokenizer: subword ids, plus <s> at the start and </s> at the end
  ▼
input_ids and attention_mask            (B, T)
  │
  ▼
roberta-base encoder, 12 layers         (B, T, 768)     124,055,040 parameters
  │  keep only the first token, <s>
  ▼
Sequence summary                        (B, 768)
  │  dropout, p = 0.1
  ▼
Linear layer, 768 → 2                   (B, 2)          1,538 parameters
  │  softmax only for confidence; argmax gives the prediction
  ▼
Prediction: 0 = negative, 1 = positive
```

**Comparison models**

- **BiLSTM:** ids (B, T) → embedding 7,005 × 128 (896,640) gives (B, T, 128) → BiLSTM with 128 units
  per direction (264,192) gives (B, T, 256) → mean and max pooling joined gives (B, 512) → dropout 0.5 →
  linear (1,026) gives (B, 2).
- **DistilBERT:** the same shape as RoBERTa, with a 6-layer encoder of 66,362,880 parameters.

**Loss:** weighted cross-entropy for all neural models. Class weights are n / (2 · n_class), about 3.4 for
negative and 0.6 for positive, so errors on the rare class cost more. The TF-IDF models use the same formula
through `class_weight='balanced'`.

**Training**

| | RoBERTa | DistilBERT | BiLSTM |
|---|---|---|---|
| Optimiser | AdamW | AdamW | AdamW |
| Learning rate | 2e-5, linear warmup for 10% of steps, then decay | Same | 1e-3, constant |
| Weight decay | 0.01, skipped on 1-D parameters | Same | 1e-4 |
| Batch size | 16 | 16 | 32 |
| Epochs | 3 | 4 | Up to 20, stops after 4 with no improvement |
| Other | fp16, gradient clip 1.0 | Same | fp16, gradient clip 1.0 |

- **Epoch kept:** the one with the best validation macro-F1.
- **Pretrained weights:** both transformers train all layers, not just the head.
- **BiLSTM start:** Xavier and orthogonal weights, and forget-gate bias 1 so it starts by remembering.

**Reproducibility:** fixed split, fixed seeds, 3-seed mean ± std. Runs were not bit-exact on the GPU: RoBERTa
seed 42 scored 0.930 and then 0.917 with identical settings. That is why the seed mean is reported.

**To include in the report:** the code and run commands, which are in `README.md`.

## (c) How the settings were chosen

**Method:** start from standard values, change one setting at a time, and repeat each value over 3 seeds, using
validation macro-F1 only. A change counts as better only if its mean gain beats the larger of the two seed standard
deviations, because 3 seeds can't separate smaller gains from luck. Files: `results/tuning/`.

**Starting values and why**

- **Transformer learning rate 2e-5:** the pretrained weights only need small adjustments.
- **2 to 4 epochs:** fine-tuning overfits quickly.
- **Maximum length 256:** covers about 98% of reviews in full.

**Results (macro-F1)**

| Model | Setting | Tested | Result | Decision |
|---|---|---|---|---|
| SVM | C | 0.1 to 30 | 0.832, 0.844, 0.849, **0.851**, 0.845, 0.843 | C=3 (gain over C=1 is only 0.002) |
| Logistic Regression | C | 0.1 to 30 | 0.770, 0.790, 0.828, 0.840, **0.844**, 0.842 | C=10 |
| DistilBERT | Learning rate | 1e-5, **2e-5**, 5e-5 | 0.893, 0.896, 0.884 | Keep; 5e-5 is too fast |
| DistilBERT | Epochs | 2, 3, **4** | 0.890, 0.892, 0.896 | Keep |
| DistilBERT | Max length | 128, 192, **256** | 0.891, 0.890, 0.896 | Keep |
| DistilBERT | Dropout | **0.1**, 0.3 | 0.896, 0.897 | Keep |
| DistilBERT | Class weighting | **On**, off | 0.896, 0.894 | Keep |
| RoBERTa | Learning rate | 1e-5, **2e-5**, 3e-5 | 0.921, 0.920, 0.922 | Keep |
| RoBERTa | Epochs | 2, **3**, 4 | 0.913, 0.926, 0.920 | 3 (see below) |
| BiLSTM | Learning rate | 3e-4, **1e-3**, 3e-3 | 0.802, 0.805, 0.809 | Keep |
| BiLSTM | Dropout | 0.2, **0.5**, 0.7 | 0.802, 0.805, 0.803 | Keep |
| BiLSTM | Hidden size | 64, **128**, 256 | 0.811, 0.805, 0.809 | Keep |
| BiLSTM | Max length | 100, **200**, 400 | 0.800, 0.805, 0.803 | Keep |
| BiLSTM | Class weighting | **On**, off | 0.805, 0.803 | Keep |

Bold marks the value used. "Keep" means no change passed the rule.

**What to say about it**

- **Most changes made no real difference.** Only RoBERTa's 3 epochs passed the rule, and it didn't hold up: the
  final rerun scored 0.922, not 0.926. Justify 3 epochs as "equal to 4 within noise, and about 25% faster", not as
  an accuracy gain.
- **Class weighting barely changed macro-F1,** despite the imbalance. It wasn't tested on RoBERTa. Don't credit it
  for handling the imbalance.
- **RoBERTa had fewer sweeps,** only learning rate and epochs, because of its running time.
- **One-at-a-time tuning has a limit:** it can't find settings that only help in combination.
- **The tuning numbers differ slightly from the main table:** DistilBERT's default scored 0.896 in the sweep and 0.894 in
  its final run, again because runs aren't bit-exact.

## (d) Test predictions

- **File:** `submission.csv`, from `make_submission.py`, using RoBERTa seed 44.
- **Checks passed:** columns `reviews,predictions`, 1,851 rows in `test.json` order, only 0 and 1.
- **Output:** 270 negative (14.6%) and 1,581 positive (85.4%).
- **Selection:** the model was chosen on validation macro-F1 before `test.json` was opened.
- **Quote 0.922 as expected performance.** This seed scored 0.929, which flatters it because it's the best of three. The
  shared validation set also makes scores slightly optimistic.
- **Overlap:** 101 test reviews (5.5%) also appear in `train.json`, which could lift the test score slightly.

## (e) Correct and incorrect test samples

The brief asks for test samples. They have no true labels, so the labels below are my judgement and your group should
agree with them first. The 15 picks are in `error_analysis.csv`. They were chosen to be hard and after seeing predictions,
so they show behaviour and **must not be reported as an accuracy**.

**Suggested cases to discuss** (row numbers count from 0 in `test.json`)

| Row | Review (shortened) | My label | RoBERTa | SVM | Point |
|---|---|---|---|---|---|
| 80 | "should really be called 'Simulation Leather'. They are warm and comfortable but... thin black plastic" | 0 | 0 (correct) | 1 | **Strength:** only RoBERTa weighs the mockery and complaint above "warm and comfortable" |
| 1612 | "Not the best fabric, Scratchy and See thru" | 0 | 0 (correct) | 1 | **Strength:** understands negation in context, where the SVM doesn't |
| 1720 | "What I love most about these gloves is that only the upper halves of my fingers are cold. What a deal" | 0 | 1 (99.9% positive) | 1 | **Weakness:** irony, with positive words throughout |
| 754 | "Bag was well made, but too small. Can't put much in it... three times bigger it would be perfect" | 0 | 1 (99.4% positive) | 1 | **Weakness:** mixed review, where positive wording wins |

**Wrong out of the 15 picks, by category**

| Category | Picks | SVM | Logistic Regression | BiLSTM | DistilBERT | RoBERTa |
|---|---|---|---|---|---|---|
| Negation | 5 | 2 | 0 | 2 | 0 | 0 |
| Sarcasm | 5 | 5 | 5 | 4 | 3 | 2 |
| Mixed sentiment | 5 | 2 | 2 | 0 | 4 | 2 |

**Evidence from labelled validation data**

- RoBERTa's 38 errors on seed 44, each given one main cause by my reading:

  | Main cause | Errors |
  |---|---|
  | Mixed sentiment | 20 |
  | Implied opinion, few sentiment words | 6 |
  | Sarcasm or humour | 4 |
  | Label looks doubtful | 4 |
  | No clear opinion | 4 |

- **Confidently wrong:** 31 of the 38 errors had over 90% confidence, so confidence wouldn't reliably flag them.
- **Shared errors:** all five models got 11 reviews wrong. Several of these are hard for any model, such as review 204,
  which reads as glowing but is labelled negative.
- **Error rates by wording:**

  | Reviews containing | Reviews | Negative share | SVM | BiLSTM | RoBERTa |
  |---|---|---|---|---|---|
  | Negation word | 561 | 21.2% | 8.7% | 13.7% | 4.3% |
  | Contrast word | 444 | 18.2% | 10.4% | 15.5% | 6.1% |
  | Neither | 383 | 5.5% | 3.7% | 5.0% | 2.1% |

  These reviews also contain more negatives, which are harder in general, so this doesn't show that the words themselves
  cause the errors.

## (f) Feature format, resources and accuracy

| Feature format | Model | Macro-F1 | Train time | Hardware | Saved model size |
|---|---|---|---|---|---|
| Sparse TF-IDF (37,543 dimensions) | Linear SVM | 0.851 | 0.49 s | CPU | Not saved |
| Learned word embeddings (128 dimensions) | BiLSTM | 0.805 | 47 s | 0.08 GB GPU | 4.8 MB |
| Contextual embeddings (768 dimensions) | DistilBERT | 0.894 | 173 s | 1.68 GB GPU | 265 MB |
| Contextual embeddings (768 dimensions) | RoBERTa | 0.922 | 263 s | 3.2–5.1 GB GPU | 496 MB |

**Points to make**

- **Cost of the best score:** RoBERTa gains 0.071 macro-F1 over the SVM, with training about 540 times slower and a GPU needed.
  Negative F1 shows the gain more clearly: 0.866 against 0.741.
- **More parameters don't guarantee better results.** The BiLSTM has 31 times the SVM's parameters and scores lower. It learned
  its embeddings from only 6,104 reviews and overfits fast: by epoch 6 the mean training loss is 0.03 and validation loss 0.89.
- **Pretrained knowledge seems to matter most:** the transformers start with language knowledge the BiLSTM must learn from scratch.
  This is an interpretation, and no experiment here isolates it.
- **DistilBERT against RoBERTa:** RoBERTa gains 0.028 macro-F1 for about 1.5 times the training time and at least twice the memory.
- **Practical choice:** if speed or a CPU-only setup matters most, TF-IDF + SVM is a strong option. Use RoBERTa when catching
  negative reviews matters and a GPU is available.
- **Timing limits:** times varied between runs (one DistilBERT job took 148 s and then 173 s), so compare orders of magnitude only.
- **Memory:** RoBERTa usually peaks at 3.2 GB, and seed 43 reached 5.1 GB. The cause is unexplained, so report a range.
- **Not measured:** pretrained word embeddings such as GloVe, and test-time prediction speed.

## Still needed for the report

- Group members and each person's contribution (required by the brief).
- Parts (g) and (h), which these notes don't cover.
- A decision on your schoolmate's BERT run. It used a different validation set, so it isn't comparable until rerun on this split.
- The module's rules on AI tools, and any disclosure they require.
- Upload both the report and `submission.csv` to NTULearn by Friday 13 Nov 2026, 11:59pm.

## References to cite

Check these details before using them.

1. Blitzer, Dredze and Pereira (2007). Biographies, Bollywood, Boom-boxes and Blenders. ACL. (dataset, from the brief)
2. Devlin et al. (2019). BERT. NAACL. (from the brief)
3. Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.
4. Sanh et al. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
5. Hochreiter and Schmidhuber (1997). Long Short-Term Memory. Neural Computation.
6. Loshchilov and Hutter (2019). Decoupled Weight Decay Regularization. ICLR. (AdamW)
7. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.
