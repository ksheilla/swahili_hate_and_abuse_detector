# Swahili Hate & Abuse Detection

> **Content warning:** this project deals with hateful and abusive language, and some examples below contain it.

A text classifier that labels Swahili social media posts as **Normal**, **Abuse** or **Hate**. It is built by fine-tuning
[AfroXLMR-base](https://huggingface.co/Davlan/afro-xlmr-base) on the Swahili part of the
[AfriHate](https://huggingface.co/datasets/afrihate/afrihate) dataset, and is deployed as a web app.

| | |
|---|---|
| **Web app** | **TODO: paste your link here → `https://YOUR-APP-NAME.streamlit.app`** |
| **Model** | [ksheilla/swahili-hate-afroxlmr](https://huggingface.co/ksheilla/swahili-hate-afroxlmr) |
| **Dataset** | [afrihate/afrihate](https://huggingface.co/datasets/afrihate/afrihate) (Swahili split) |
| **Notebook** | [`notebooks/swahili_hate_detector.ipynb`]|
| **Test macro-F1** | **0.899** (TF-IDF baseline: 0.877) |

---

## 1. Problem

Swahili is spoken by tens of millions of people in East Africa and is widely used on social media, often mixed with English
and Sheng. Most moderation tools are built for English and miss hate speech written in Swahili. This project builds a model
that flags abusive and hateful Swahili posts so that **human moderators** can review them faster. It is meant to assist
moderators, not to remove content automatically.

## 2. Dataset

**Source.** AfriHate ([Muhammad et al., 2025](https://aclanthology.org/2025.naacl-long.92/)) is a collection of tweets
in 15 African languages, posted between 2012 and 2023. The tweets were annotated by native speakers and the final label
was chosen by majority vote. The dataset is released under the Apache-2.0 license. The paper defines the classes as follows:

| Label | Definition (from the AfriHate paper) |
|---|---|
| **Hate** | Expresses hatred towards a group or individual based on political affiliation, race, ethnicity, religion, gender, sexual orientation or other characteristics, including threats of violence. |
| **Abuse** | Uses words inappropriately, e.g. swearing, name-calling or profanity, without targeting such a characteristic. |
| **Normal** | Neither hateful nor abusive. |

**Size and balance.** The Swahili split has 14,760 / 3,164 / 3,168 tweets (train / dev / test). Tweets are short, with a
median of 12 words. The classes are imbalanced: 44.6% Normal, 36.5% Abuse and 18.8% Hate.

**Quality problems found** (notebook section 2):

| Problem | Train | Dev | Test |
|---|---|---|---|
| Duplicate tweets within the split | 838 | 37 | 41 |
| Same tweet with different labels | 31 | 3 | 2 |
| Tweets shared with another split | 347 with dev, 387 with test | 71 with test | |

Annotator agreement for Swahili is moderate (free-marginal kappa 0.55, reported in the paper), so some labels are noisy.

**Preprocessing** (notebook section 3). Tweets were matched after lower-casing and collapsing spaces. Then:

1. Tweets with conflicting labels were dropped.
2. One copy of each duplicate was kept.
3. Dev tweets that also appear in train were removed.
4. Test tweets that also appear in train or dev were removed.

The cleaned splits have **13,876 / 2,766 / 2,664** tweets, with a similar class balance. The text itself was not changed
further, because the tokenizer handles casing and punctuation.

## 3. Method

**Baseline.** TF-IDF features over word unigrams and bigrams (min_df 2, sublinear tf), fed to a logistic regression with
balanced class weights. It is fast and easy to interpret, and it shows how far surface word cues alone can go.

**Main model.** AfroXLMR-base ([Alabi et al., 2022](https://aclanthology.org/2022.coling-1.382/)) is XLM-RoBERTa-base
([Conneau et al., 2020](https://aclanthology.org/2020.acl-main.747/)) with further masked-language-model pre-training on
17 African languages, including Swahili. It has about 278M parameters.

- **Input:** a tweet, split into subword tokens by a SentencePiece tokenizer (250k vocabulary) and truncated to 64 tokens.
- **Encoder:** a 12-layer Transformer encoder (hidden size 768, 12 attention heads) produces a contextual embedding for every token.
- **Output:** a classification head (dense → tanh → linear) reads the `<s>` token's embedding and outputs 3 logits. Softmax turns them into probabilities for Abuse, Hate and Normal.
- **Fine-tuning:** all weights are updated with cross-entropy loss and AdamW. The best epoch is chosen by dev macro-F1.

| Hyperparameter | Value |
|---|---|
| Learning rate | 2e-5, linear decay |
| Warm-up | 350 steps (about 10% of training) |
| Batch size | 16 (train), 64 (eval) |
| Epochs | 4, keeping the best checkpoint by dev macro-F1 |
| Weight decay | 0.01 |
| Max sequence length | 64 tokens |
| Precision | fp16 mixed precision on GPU |
| Seed | 42 |

**Metrics.** The main metric is macro-F1, the average F1 over the three classes. Because each class counts equally, the
minority Hate class weighs as much as Normal, which plain accuracy does not do. Accuracy, per-class precision and recall,
and confusion matrices are also reported.

## 4. Experiments

All models are trained on the cleaned train set, tuned on dev, and scored once on the cleaned test set. The full table and
discussion are in [`results/experiments.md`](results/experiments.md).

| Model | Seed | Dev macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|---|
| TF-IDF + logistic regression (baseline) | – | 0.8760 | 0.8855 | 0.8765 |
| **AfroXLMR-base (deployed)** | 42 | 0.8939 | **0.9073** | **0.8992** |
| AfroXLMR-base | 123 | 0.8952 | 0.9017 | 0.8917 |
| XLM-R base | 42 | 0.8961 | 0.9024 | 0.8922 |

What these results show:

- **Fine-tuned Transformers beat the baseline** by 1.5–2.3 macro-F1 points. Most of the gain is on the classes that
  depend on context: Hate recall rises from 0.804 to 0.842, and Abuse recall from 0.891 to 0.937.
- **The baseline is strong**, because slurs and profanity are powerful word-level cues. That is why the gap is modest.
- **AfroXLMR and XLM-R cannot be separated.** The difference between two AfroXLMR seeds (0.0075) is as large as the gap
  between the models (0.0070), and XLM-R is slightly ahead on dev. One plausible reason is that Swahili was already part of
  XLM-R's pre-training data, so African-language adaptation adds less than it would for a lower-resource language.
- **Leakage inflates scores.** When the baseline is trained on the raw data, it gets 97.4% accuracy on test tweets that
  also appear in train, against 88.9% on unseen ones. Its macro-F1 drops from 0.896 on the raw test set to 0.879 on the
  cleaned one.

The AfriHate paper reports 88.0 macro-F1 for Swahili (monolingual AfroXLMR-76L) and 89.5 (multilingual training), measured
on the original test set. Our scores are on the cleaned test set, so they are not directly comparable.

## 5. Results of the deployed model (test set, 2,664 tweets)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Abuse | 0.899 | 0.937 | 0.918 | 1,135 |
| Hate | 0.899 | 0.842 | 0.869 | 423 |
| Normal | 0.919 | 0.901 | 0.910 | 1,106 |
| **Macro average** | **0.906** | **0.893** | **0.899** | 2,664 |

Confusion matrix (rows = true label, columns = predicted label):

| | → Abuse | → Hate | → Normal |
|---|---|---|---|
| **Abuse** | 1,064 | 14 | 57 |
| **Hate** | 36 | 356 | 31 |
| **Normal** | 83 | 26 | 997 |

## 6. Error analysis

The error analysis (notebook section 7) runs the saved model over the test set in float32. That flips one borderline
tweet compared with the table above, giving 248 errors (9.3%). The most common errors are Normal → Abuse (83),
Abuse → Normal (58), Hate → Abuse (36) and Hate → Normal (31). Reading samples of each type shows five patterns:

1. **Profanity in tweets labelled Normal** produces false Abuse alarms. Several of these tweets contain explicit slurs or
   swearing, so given the moderate annotator agreement, some of them look like label disagreements rather than model mistakes.
2. **Implicit hate without slurs is missed.** For example, *"Ni kama wanawake hawakuumbwa na mungu"* ("It's as if women
   were not created by God") is labelled Hate, but the model predicts Normal with 97% confidence. Without an obvious
   offensive word, the model has little to go on.
3. **Hate and Abuse get confused** when a tweet contains both profanity and an identity target. The model tends to
   follow the profanity and predict Abuse.
4. **Mild insults are uncertain.** *"mtu mwambie huyu bwege kwa stfu"* ("someone tell this fool to stfu") is Abuse,
   but the model predicts Normal with only 55% confidence.
5. **Code-switching and noisy text.** English slang, usernames and hashtags fused with Swahili words are common in the
   errors, and several misclassified tweets read like word-for-word translations of English tweets.

**Confidence is informative but not enough on its own.** Tweets predicted with more than 99% confidence are 97.3%
correct, and they make up 72% of the test set. However, 62% of all errors still have more than 90% confidence. Sending
posts below **95% confidence** to a human flags 11.4% of posts and catches 52.8% of the errors. The web app uses this
threshold.

## 7. Web app

[`app.py`](app.py) is a Streamlit app deployed on Streamlit Community Cloud straight from this repository.

1. On start-up, it downloads the tokenizer and the fine-tuned model from the Hugging Face Hub and keeps them in memory
   (`st.cache_resource`).
2. The user types or picks a Swahili post.
3. The text goes through the same tokenizer (truncated to 64 tokens) and the model, and a softmax over the 3 logits gives
   the class probabilities.
4. The app shows the predicted label, a bar for each class's probability, and a moderation note. The note asks for human
   review whenever confidence is below 95%.

Loading the model and making a prediction takes about 1.6 GB of RAM (the notebook's memory check), which is within the
free tier's limit of about 2.7 GB. The app has three example buttons: a neutral greeting, and two test-set posts that the
model gets wrong (one with low confidence, one with high confidence).

## 8. How to reproduce

**Training and evaluation** (Google Colab with a GPU runtime):

1. Open `notebooks/swahili_hate_afroxlmr.ipynb` in Colab.
2. Add Colab secrets `HF_TOKEN` (a read token) and `HF_WRITE_TOKEN` (a write token, only needed to upload the model).
3. Run the cells from top to bottom. The notebook downloads AfriHate itself, then saves the model and the cleaned splits
   to `MyDrive/swahili-hate-afroxlmr` on Google Drive. The library versions are listed in `requirements-train.txt`.

**Run the web app locally:**

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
pip install -r requirements.txt
streamlit run app.py
```

**Deploy your own copy:** at [share.streamlit.io](https://share.streamlit.io), click **Create app** and choose this
repository with `app.py` as the file. Under **Advanced settings**, pick Python 3.12.

## 9. Repository structure

```
├── README.md                     this file
├── app.py                        Streamlit web app
├── requirements.txt              web app dependencies (used by Streamlit Community Cloud)
├── requirements-train.txt        notebook dependencies
├── notebook/
│   └── swahili_hate_detector.ipynb   data preparation, training, experiments, error analysis, deployment
└── results/
    └── experiments.md            experiment table and discussion
```

The model weights are stored on the Hugging Face Hub, not in this repository, and the dataset is downloaded by the notebook.

## 10. Limitations and ethical considerations

- **Noisy labels.** Swahili annotator agreement is moderate (kappa 0.55), so the test labels themselves contain disagreements.
- **Domain.** The data comes from Twitter between 2012 and 2023, and many of the tweets we inspected discuss Kenyan
  politics. Performance may drop on other platforms, other regions, newer slang or long texts (input is cut at 64 tokens).
- **Missed hate is the costliest error.** Hate is the smallest class and has the lowest recall (0.842). Implicit hate
  without slurs is the hardest case.
- **Few runs.** Most configurations were trained once, so differences below about one macro-F1 point should not be
  treated as real.
- **Use.** The model can be biased, for example towards flagging profanity or reclaimed words. It should support human
  moderators and must not be used for automated removal or to make decisions about individuals.

## References

- Muhammad, S. H., Abdulmumin, I., Ayele, A. A., Adelani, D. I., Ahmad, I. S., et al. (2025). AfriHate: A Multilingual
  Collection of Hate Speech and Abusive Language Datasets for African Languages. *Proceedings of NAACL 2025*, 1854–1871.
  https://aclanthology.org/2025.naacl-long.92/
- Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting Pre-trained Language Models to African
  Languages via Multilingual Adaptive Fine-Tuning. *Proceedings of COLING 2022*, 4336–4349.
  https://aclanthology.org/2022.coling-1.382/
- Conneau, A., Khandelwal, K., Goyal, N., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale.
  *Proceedings of ACL 2020*, 8440–8451. https://aclanthology.org/2020.acl-main.747/
