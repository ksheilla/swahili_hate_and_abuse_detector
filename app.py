import os

import streamlit as st
import torch

MODEL_ID = os.environ.get("MODEL_ID", "ksheilla/swahili-hate-afroxlmr")
MAX_LEN = 64              # same max_length used during fine-tuning
REVIEW_THRESHOLD = 0.95   # from the confidence analysis: below this, a human should review

LABELS = ["Normal", "Abuse", "Hate"]  # fixed display order for the probability bars
# Class colours, checked for contrast and colour-blind separation on the dark background
LABEL_COLOR = {"Normal": "#1C9DBA", "Abuse": "#BF7F1F", "Hate": "#E8487E"}
LABEL_HELP = {
    "Normal": "No abusive or hateful content detected.",
    "Abuse": "Offensive or insulting language that does not target a group identity.",
    "Hate": "Attacks a person or group because of an identity such as ethnicity, religion or gender.",
}

# (button label, tooltip, text)
EXAMPLES = [
    ("Greeting", "Written for the demo", "Habari za asubuhi, natumai una siku njema."),
    ("Mild insult", "AfriHate test set, true label: Abuse", "mtu mwambie huyu bwege kwa stfu"),
    ("Implicit hate", "AfriHate test set, true label: Hate", "Ni kama wanawake hawakuumbwa na mungu"),
]

# Test-set results of the deployed model (see the notebook): precision, recall, F1
TEST_RESULTS = {"Normal": (0.919, 0.901, 0.910), "Abuse": (0.899, 0.937, 0.918), "Hate": (0.899, 0.842, 0.869)}

st.set_page_config(page_title="Swahili Hate & Abuse Detector", page_icon="🛡️")

# Colours and font come from .streamlit/config.toml; this adds the cards, tiles and bars.
CSS = """
:root { --panel: #1E2142; --raised: #262A52; --line: #2D3163; --ink: #F2F3FF; --muted: #9EA3CC; }
.block-container { padding-top: 4.5rem; }
.msr {
  font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; font-size: 20px;
  line-height: 1; letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap;
  direction: ltr; font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased; vertical-align: middle;
}

/* header */
.brand { display: flex; align-items: center; gap: 14px; margin-bottom: 4px; }
.logo {
  width: 46px; height: 46px; border-radius: 14px; flex: none; display: grid; place-items: center; color: #fff;
  background: linear-gradient(135deg, #3B3FE0, #8A4DFF 60%, #E8487E);
}
.logo .msr { font-size: 26px; }
.app-name { font-size: 1.35rem; font-weight: 600; color: var(--ink); line-height: 1.2; }
.greeting { color: var(--muted); font-size: 0.92rem; margin-top: 2px; }
.warning-line { display: flex; align-items: flex-start; gap: 6px; color: var(--muted); font-size: 0.78rem; line-height: 1.45; margin-top: 6px; }
.warning-line .msr { color: #BF7F1F; font-size: 16px; margin-top: 1px; }

.section-title { font-size: 1.1rem; font-weight: 600; color: var(--ink); margin: 18px 0 12px; }
.section-title.tight { margin-bottom: 0; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 16px 18px; }

/* score tiles */
.tiles { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.tile { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 14px; min-width: 0; }
.tile-label { display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 0.78rem; white-space: nowrap; }
.tile-label i { width: 3px; height: 14px; border-radius: 2px; display: inline-block; }
.tile-value { font-size: 1.55rem; font-weight: 600; color: var(--ink); margin: 8px 0 2px; font-variant-numeric: tabular-nums; }
.tile-sub { font-size: 0.74rem; color: var(--muted); }
.tile-sub b { color: #3CC59A; font-weight: 600; }

/* "Check a post" panel (a Streamlit container with key="check_panel") */
.st-key-check_panel { background: var(--panel); border: 1px solid var(--line); border-radius: 20px; padding: 18px; }
.st-key-check_panel [data-testid="stBaseButton-secondary"] {
  background: var(--raised); border: 1px solid var(--line); color: var(--ink); font-size: 0.85rem;
}
.st-key-check_panel [data-testid="stBaseButton-secondary"]:hover { border-color: #A796FF; color: #fff; }
.st-key-check_panel [data-testid="stBaseButton-primary"] { font-weight: 600; min-height: 2.75rem; }

/* prediction card and probability bars */
.pred-card {
  border-radius: 20px; padding: 20px 22px; color: #fff;
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
.bars { margin-top: 12px; }
.bar-row { margin-bottom: 12px; }
.bar-row:last-child { margin-bottom: 0; }
.bar-head { display: flex; justify-content: space-between; font-size: 0.88rem; color: var(--ink); margin-bottom: 5px; }
.bar-head.top { font-weight: 600; }
.bar-track { height: 8px; border-radius: 4px; background: var(--raised); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; }

/* results table */
.table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
.table th { text-align: left; color: var(--muted); font-weight: 500; font-size: 0.76rem; padding: 0 0 10px; }
.table td { padding: 10px 0; border-top: 1px solid var(--line); color: var(--ink); font-variant-numeric: tabular-nums; }
.table th:not(:first-child), .table td:not(:first-child) { text-align: right; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 9px; vertical-align: middle; }

@media (max-width: 640px) {
  .brand { align-items: flex-start; }
  .tiles { grid-template-columns: 1fr; }
}
"""


def icon(name):
    return f'<span class="msr" aria-hidden="true">{name}</span>'


def show(markup):
    """Render a block of HTML. Lines are stripped so indentation in the source doesn't matter."""
    st.html(" ".join(line.strip() for line in markup.splitlines() if line.strip()))


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

show(f"""
<div class="brand">
  <div class="logo">{icon("shield")}</div>
  <div>
    <div class="app-name">Swahili Hate &amp; Abuse Detector</div>
    <div class="greeting">Habari! Paste a Swahili post to check it for abusive or hateful language.</div>
  </div>
</div>
<div class="warning-line">{icon("warning")}<span>Content warning: shows hateful and abusive language. Built to assist
human moderators, not to remove posts automatically.</span></div>
""")

tokenizer, model = load_model()

show("""
<div class="section-title">Overview</div>
<div class="tiles">
  <div class="tile">
    <div class="tile-label"><i style="background:#6D4EF5"></i>Macro-F1</div>
    <div class="tile-value">0.899</div>
    <div class="tile-sub"><b>+0.023</b> over the TF-IDF baseline</div>
  </div>
  <div class="tile">
    <div class="tile-label"><i style="background:#1C9DBA"></i>Accuracy</div>
    <div class="tile-value">90.7%</div>
    <div class="tile-sub">on 2,664 unseen test tweets</div>
  </div>
  <div class="tile">
    <div class="tile-label"><i style="background:#E8487E"></i>Hate recall</div>
    <div class="tile-value">84.2%</div>
    <div class="tile-sub">of hateful test posts caught</div>
  </div>
</div>
<div class="section-title tight">Check a post</div>
""")

with st.container(key="check_panel"):
    st.caption("Try an example")
    for i, (col, (label, tip, text)) in enumerate(zip(st.columns(len(EXAMPLES)), EXAMPLES)):
        col.button(label, help=tip, on_click=use_example, args=(text,), width="stretch", key=f"example_{i}")
    text = st.text_area("Swahili post", key="text", height=120, placeholder="Andika chapisho hapa...")
    clicked = st.button("Classify post", type="primary", key="classify", width="stretch")
    if clicked and not text.strip():
        st.warning("Please enter some Swahili text.")

if clicked and text.strip():
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
    show(f"""
    <div class="section-title">Result</div>
    <div class="pred-card">
      <div class="pc-top">
        <div><div class="pc-caption">Prediction</div><div class="pc-label">{label}</div></div>
        <div class="pc-chip"><i style="background:{LABEL_COLOR[label]}"></i>{conf:.1%}</div>
      </div>
      <div class="pc-help">{LABEL_HELP[label]}</div>
    </div>
    <div class="panel bars">{bars}</div>
    """)
    if conf < REVIEW_THRESHOLD:
        st.warning(note)
    else:
        st.info(note)

rows = "".join(
    f'<tr><td><span class="dot" style="background:{LABEL_COLOR[name]}"></span>{name}</td>'
    f"<td>{p:.3f}</td><td>{r:.3f}</td><td>{f:.3f}</td></tr>"
    for name, (p, r, f) in TEST_RESULTS.items()
)
show(f"""
<div class="section-title">Test-set results by class</div>
<div class="panel"><table class="table">
  <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
  <tbody>{rows}</tbody>
</table></div>
""")

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
