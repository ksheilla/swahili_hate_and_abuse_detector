# Experiments

> Content warning: the error examples quote abusive and hateful language from the dataset.

All numbers come from the notebook (`notebooks/swahili_hate_afroxlmr.ipynb`). Unless stated otherwise, models are trained
on the **cleaned** train set (13,876 tweets), the checkpoint is chosen on the cleaned dev set (2,766), and each model is
scored once on the cleaned test set (2,664). The main metric is **macro-F1**, which gives the three classes equal weight,
so the minority Hate class (about 17%) matters as much as Normal.

## 1. Model comparison

| # | Model | What changes from the previous row | Seed | Dev macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|---|---|---|
| E1 | TF-IDF (1–2-grams) + logistic regression, balanced class weights | Baseline | – | 0.8760 | 0.8855 | 0.8765 |
| E2 | AfroXLMR-base, fine-tuned (**deployed**) | Pre-trained Transformer instead of word counts | 42 | 0.8939 | 0.9073 | 0.8992 |
| E3 | XLM-R base, fine-tuned | Same recipe, without the African-language pre-training | 42 | 0.8961 | 0.9024 | 0.8922 |
| E4 | AfroXLMR-base, fine-tuned | Same as E2, different random seed | 123 | 0.8952 | 0.9017 | 0.8917 |

Shared settings for E2–E4: learning rate 2e-5, batch size 16, 4 epochs, weight decay 0.01, max length 64, fp16,
best epoch chosen by dev macro-F1. Warm-up is 350 steps for E2 and 346 steps (10% of training) for E3 and E4.

### Baseline vs. deployed model, per class (test set)

| Class | Baseline P / R / F1 | AfroXLMR (E2) P / R / F1 | F1 change |
|---|---|---|---|
| Abuse | 0.905 / 0.891 / 0.898 | 0.899 / 0.937 / 0.918 | +0.020 |
| Hate | 0.888 / 0.804 / 0.844 | 0.899 / 0.842 / 0.869 | +0.025 |
| Normal | 0.866 / 0.911 / 0.888 | 0.919 / 0.901 / 0.910 | +0.022 |

The baseline makes 305 test errors and AfroXLMR makes 247, which is 19% fewer.

## 2. Leakage experiment

**Question:** the raw splits share tweets (387 tweets appear in both train and test). How much does that inflate the
scores? To find out, the baseline (E1 settings) is trained on the **raw** train set and scored on four test sets.

| Test set | Tweets | Accuracy | Macro-F1 |
|---|---|---|---|
| Raw test set | 3,168 | 0.8999 | 0.8959 |
| Cleaned test set | 2,664 | 0.8874 | 0.8785 |
| Only test tweets that also occur in train ("leaked") | 391 | **0.9744** | **0.9631** |
| Only test tweets not in train ("fresh") | 2,777 | 0.8894 | 0.8819 |

The leaked count (391) is higher than 387 because matching ignores case and extra spaces.

## 3. What we learned

1. **Pre-trained Transformers beat the baseline, but by a modest margin** (+1.5 to +2.3 macro-F1 points). The gains are
   largest where context matters. Hate recall rises from 0.804 to 0.842, so fewer hateful posts are missed. Abuse recall
   rises from 0.891 to 0.937. The baseline is strong because slurs and profanity are reliable word-level cues for
   Abuse and much Hate.
2. **AfroXLMR vs. XLM-R is not conclusive.** Changing only the seed moves AfroXLMR's test macro-F1 by 0.0075 (E2 vs. E4).
   That is about the same as the gap between AfroXLMR and XLM-R (0.0070, E2 vs. E3), and XLM-R is slightly ahead on dev.
   Averaged over its two seeds, AfroXLMR scores 0.8955, close to XLM-R's 0.8922. A plausible explanation is that Swahili
   is already among the 100 languages XLM-R was pre-trained on, so adapting it to African languages adds less than it
   would for a language XLM-R never saw. Settling the question would need at least 3–5 seeds per model.
3. **The deployed model's test score is slightly optimistic.** E2 was trained and saved before the other runs, and it
   turned out to be the best of three runs on test. The mean of the AfroXLMR runs (0.8955) is a fairer estimate of what
   this recipe achieves.
4. **Leakage matters.** The model gets leaked test tweets right 97.4% of the time, against 88.9% for fresh tweets, so
   leakage adds about 1.7 macro-F1 points (0.8959 vs. 0.8785). This is why every model here is scored on the cleaned
   test set. For context, the AfriHate paper reports 88.0 (monolingual) and 89.5 (multilingual) macro-F1 for Swahili
   with AfroXLMR-76L on the original test set, so those numbers are not directly comparable with ours.

## 4. Error analysis of the deployed model

Re-running the saved model in float32 gives 248 errors out of 2,664 (one borderline tweet flips compared with the fp16
evaluation above).

| True → Predicted | Count |
|---|---|
| Normal → Abuse | 83 |
| Abuse → Normal | 58 |
| Hate → Abuse | 36 |
| Hate → Normal | 31 |
| Normal → Hate | 26 |
| Abuse → Hate | 14 |

Error types seen when reading samples:

- **Possible label noise.** Some tweets labelled Normal contain explicit slurs or sexual swearing, and the model calls
  them Abuse. With annotator agreement of kappa 0.55, part of the Normal → Abuse errors are likely labelling disagreements.
- **Implicit hate.** Hate expressed without offensive words is missed with high confidence, for example *"Ni kama
  wanawake hawakuumbwa na mungu"* ("It's as if women were not created by God"), which is predicted Normal at 97%.
- **Hate vs. Abuse.** Tweets that contain both profanity and an identity target are often predicted as Abuse, because
  profanity is the stronger surface cue.
- **Mild or rare insults.** *"mtu mwambie huyu bwege kwa stfu"* ("someone tell this fool to stfu") is Abuse but is
  predicted Normal at 55% confidence.
- **Code-switching and noisy text.** English slang, usernames and hashtags glued onto Swahili words are common among the
  errors, and some misclassified tweets read like word-for-word translations of English tweets.

### Confidence and the review threshold

| Confidence | Tweets | Accuracy | Errors |
|---|---|---|---|
| ≤ 0.60 | 37 | 0.514 | 18 |
| 0.60–0.80 | 93 | 0.559 | 41 |
| 0.80–0.90 | 83 | 0.578 | 35 |
| 0.90–0.95 | 96 | 0.594 | 39 |
| 0.95–0.99 | 441 | 0.855 | 64 |
| > 0.99 | 1,914 | 0.973 | 51 |

| Send to human review if confidence < | Posts flagged | Errors caught |
|---|---|---|
| 0.80 | 4.9% | 23.8% |
| 0.90 | 8.0% | 37.5% |
| **0.95 (used in the app)** | **11.4%** | **52.8%** |

Low confidence is a strong warning sign: below 95% confidence the model is right only about 57% of the time. But
62% of errors (154 of 248) still have more than 90% confidence, so a threshold alone cannot catch every mistake.

## 5. Next steps

- Run 3–5 seeds per model and report the mean and standard deviation.
- Try class-weighted loss or threshold tuning for the Hate class, to raise Hate recall.
- Re-check a sample of the confident errors with a Swahili speaker to separate model errors from label noise.
