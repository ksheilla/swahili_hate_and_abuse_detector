import os

import streamlit as st
import torch

MODEL_ID = os.environ.get("MODEL_ID", "ksheilla/swahili-hate-afroxlmr")
MAX_LEN = 64              # same max_length used during fine-tuning
REVIEW_THRESHOLD = 0.95   # from the confidence analysis: below this, a human should review

LABEL_HELP = {
    "Normal": "No abusive or hateful content detected.",
    "Abuse": "Offensive or insulting language that does not target a group identity.",
    "Hate": "Attacks a person or group because of an identity such as ethnicity, religion or gender.",
}
LABEL_COLOR = {"Normal": "green", "Abuse": "orange", "Hate": "red"}

# (button label, tooltip, text)
EXAMPLES = [
    ("Greeting", "Written for the demo", "Habari za asubuhi, natumai una siku njema."),
    ("Test set: Abuse", "AfriHate test set, true label: Abuse", "mtu mwambie huyu bwege kwa stfu"),
    ("Test set: Hate", "AfriHate test set, true label: Hate", "Ni kama wanawake hawakuumbwa na mungu"),
]

st.set_page_config(page_title="Swahili Hate & Abuse Detector", page_icon="🛡️")


@st.cache_resource(show_spinner="Loading the model (the first load takes about a minute)...")
def load_model():
    """Download the tokenizer and fine-tuned model once and keep them in memory."""
    # Imported here rather than at the top of the file: when several visitors open the app at the
    # same moment, importing transformers in parallel can fail with "cannot import name ...".
    # st.cache_resource runs this function once and makes the other sessions wait for it.
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


def classify(text, tokenizer, model):
    """Return {label: probability} for one post."""
    inputs = tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits          # shape (1, 3): one score per class
    probs = torch.softmax(logits, dim=-1)[0]     # scores -> probabilities that sum to 1
    return {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}


def use_example(text):
    st.session_state.text = text


st.title("🛡️ Swahili Hate & Abuse Detector")
st.write("Paste a Swahili social media post to classify it as **Normal**, **Abuse** or **Hate**.")
st.caption(
    "⚠️ Content warning: this tool deals with hateful and abusive language. "
    "It is an assistive tool for human moderators and must not be used for automated removal."
)

tokenizer, model = load_model()

st.write("Try an example:")
for col, (label, tip, text) in zip(st.columns(len(EXAMPLES)), EXAMPLES):
    col.button(label, help=tip, on_click=use_example, args=(text,), width="stretch")

text = st.text_area("Swahili post", key="text", height=120, placeholder="Andika chapisho hapa...")

if st.button("Classify", type="primary", key="classify"):
    if not text.strip():
        st.warning("Please enter some Swahili text.")
    else:
        probs = classify(text, tokenizer, model)
        label, conf = max(probs.items(), key=lambda kv: kv[1])

        st.subheader(f"Prediction: :{LABEL_COLOR[label]}[{label}]")
        st.caption(LABEL_HELP[label])
        for name, p in sorted(probs.items(), key=lambda kv: -kv[1]):
            st.progress(p, text=f"{name}: {p:.1%}")

        if conf < REVIEW_THRESHOLD:
            st.warning(f"Low confidence ({conf:.0%}): send to a human moderator for review.")
        else:
            st.info(f"Confidence {conf:.0%}. A human should still review any flagged post.")

with st.expander("How it works"):
    st.markdown(
        f"""
- **Model:** [AfroXLMR-base](https://huggingface.co/Davlan/afro-xlmr-base) (XLM-RoBERTa adapted to
  African languages), fine-tuned for 3-class classification on the Swahili split of
  [AfriHate](https://huggingface.co/datasets/afrihate/afrihate). Weights:
  [{MODEL_ID}](https://huggingface.co/{MODEL_ID}).
- **Pipeline:** text → SentencePiece tokenizer (max {MAX_LEN} tokens) → 12-layer Transformer encoder →
  classification head on the `<s>` token → softmax over Abuse / Hate / Normal.
- **Data:** 13,876 training tweets after removing duplicates, conflicting labels and train/test overlap.
- **Test results:** accuracy 0.907, macro-F1 0.899. TF-IDF + logistic regression baseline: 0.886 / 0.877.
- **Review threshold ({REVIEW_THRESHOLD:.0%}):** on the test set, posts below this confidence are 11.4% of
  all posts but contain 52.8% of the model's errors.
"""
    )
