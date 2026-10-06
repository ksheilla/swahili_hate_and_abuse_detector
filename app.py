import os

import streamlit as st
import torch

MODEL_ID = os.environ.get("MODEL_ID", "ksheilla/swahili-hate-afroxlmr")
MAX_LEN = 64              # same max_length used during fine-tuning
REVIEW_THRESHOLD = 0.95   # from the confidence analysis: below this, a human should review

LABELS = ["Normal", "Abuse", "Hate"]  # fixed display order for the probability bars
LABEL_COLOR = {"Normal": "#1C9DBA", "Abuse": "#BF7F1F", "Hate": "#E8487E"}
LABEL_HELP = {
    "Normal": "No abusive or hateful content detected.",
    "Abuse": "Offensive or insulting language that does not target a group identity.",
    "Hate": "Attacks a person or group because of an identity such as ethnicity, religion or gender.",
}

# (button label, tooltip, text)
EXAMPLES = [
    ("Greeting", "Written for the demo", "Habari za asubuhi, natumai una siku njema."),
    ("Test set: Abuse", "AfriHate test set, true label: Abuse", "mtu mwambie huyu bwege kwa stfu"),
    ("Test set: Hate", "AfriHate test set, true label: Hate", "Ni kama wanawake hawakuumbwa na mungu"),
]

st.set_page_config(page_title="Swahili Hate & Abuse Detector", page_icon="🛡️")

# Prediction card and probability bars (colours and font come from .streamlit/config.toml)
CSS = """
.pred-card {
  border-radius: 20px; padding: 20px 22px; margin: 6px 0 4px; color: #fff;
  background:
    radial-gradient(circle at 90% 95%, rgba(52, 225, 163, 0.8) 0, rgba(52, 225, 163, 0) 45%),
    radial-gradient(circle at 75% 15%, rgba(255, 92, 168, 0.7) 0, rgba(255, 92, 168, 0) 40%),
    linear-gradient(135deg, #1F2BB8 0%, #4B35E8 50%, #8C4BFF 100%);
}
.pc-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
.pc-caption { font-size: 0.82rem; opacity: 0.85; }
.pc-label { font-size: 2rem; font-weight: 700; line-height: 1.15; }
.pc-chip {
  display: flex; align-items: center; gap: 6px; padding: 5px 10px; border-radius: 999px; font-size: 0.85rem;
  font-weight: 600; white-space: nowrap; background: rgba(255, 255, 255, 0.16); border: 1px solid rgba(255, 255, 255, 0.3);
}
.pc-chip i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.7); }
.pc-help { font-size: 0.88rem; opacity: 0.92; margin-top: 14px; }
.bar-row { margin-top: 12px; }
.bar-head { display: flex; justify-content: space-between; font-size: 0.88rem; margin-bottom: 5px; }
.bar-head.top { font-weight: 600; }
.bar-track { height: 8px; border-radius: 4px; background: #262A52; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; }
"""


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


st.html(f"<style>{CSS}</style>")
st.session_state.setdefault("last_result", None)

st.title("🛡️ Swahili Hate & Abuse Detector")
st.write("Paste a Swahili social media post to classify it as **Normal**, **Abuse** or **Hate**.")
st.caption(
    "CONTENT WARNING: this tool deals with hateful and abusive language. "
    "It is an assistive tool for human moderators and must not be used for automated removal."
)

tokenizer, model = load_model()

st.write("Try an example:")
for i, (col, (label, tip, text)) in enumerate(zip(st.columns(len(EXAMPLES)), EXAMPLES)):
    col.button(label, help=tip, on_click=use_example, args=(text,), width="stretch", key=f"example_{i}")

text = st.text_area("Swahili post", key="text", height=120, placeholder="Andika chapisho hapa...")

if st.button("Classify", type="primary", key="classify"):
    if not text.strip():
        st.warning("Please enter some Swahili text.")
    else:
        probs = classify(text, tokenizer, model)
        label, conf = max(probs.items(), key=lambda kv: kv[1])
        if conf < REVIEW_THRESHOLD:
            note = f"Low confidence ({conf:.0%}): send to a human moderator for review."
        else:
            note = f"Confidence {conf:.0%}. A human should still review any flagged post."
        st.session_state.last_result = {"text": text, "label": label, "conf": conf, "probs": probs, "note": note}

        bars = "".join(
            f'<div class="bar-row"><div class="bar-head{" top" if name == label else ""}">'
            f'<span>{name}</span><span>{probs[name]:.1%}</span></div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{probs[name] * 100:.1f}%;'
            f'background:{LABEL_COLOR[name]}"></div></div></div>'
            for name in LABELS
        )
        st.html(
            f'<div class="pred-card"><div class="pc-top">'
            f'<div><div class="pc-caption">Prediction</div><div class="pc-label">{label}</div></div>'
            f'<div class="pc-chip"><i style="background:{LABEL_COLOR[label]}"></i>{conf:.1%}</div></div>'
            f'<div class="pc-help">{LABEL_HELP[label]}</div></div>{bars}'
        )
        if conf < REVIEW_THRESHOLD:
            st.warning(note)
        else:
            st.info(note)

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
