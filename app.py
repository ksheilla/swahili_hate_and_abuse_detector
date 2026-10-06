import html
import os

import streamlit as st
import torch

MODEL_ID = os.environ.get("MODEL_ID", "ksheilla/swahili-hate-afroxlmr")
MAX_LEN = 64              # same max_length used during fine-tuning
REVIEW_THRESHOLD = 0.95   # from the confidence analysis: below this, a human should review
HISTORY_SIZE = 6

LABELS = ["Normal", "Abuse", "Hate"]  # fixed display order
# Class colours, checked for contrast and colour-blind separation on the dark panels
LABEL_COLOR = {"Normal": "#1C9DBA", "Abuse": "#BF7F1F", "Hate": "#E8487E"}
LABEL_ICON = {"Normal": "check_circle", "Abuse": "warning", "Hate": "report"}
LABEL_HELP = {
    "Normal": "No abusive or hateful content detected.",
    "Abuse": "Offensive or insulting language that does not target a group identity.",
    "Hate": "Attacks a person or group because of an identity such as ethnicity, religion or gender.",
}

# (button label, tooltip, text)
EXAMPLES = [
    ("Greeting", "Written for this demo", "Habari za asubuhi, natumai una siku njema."),
    ("Mild insult", "From the AfriHate test set. True label: Abuse", "mtu mwambie huyu bwege kwa stfu"),
    ("Implicit hate", "From the AfriHate test set. True label: Hate", "Ni kama wanawake hawakuumbwa na mungu"),
]

# Test-set results of the deployed model (see the notebook): precision, recall, F1
TEST_RESULTS = {"Normal": (0.919, 0.901, 0.910), "Abuse": (0.899, 0.937, 0.918), "Hate": (0.899, 0.842, 0.869)}

st.set_page_config(page_title="Swahili Hate & Abuse Detector", layout="wide")

CSS = """
:root {
  --page: #15172F; --panel: #1E2142; --raised: #262A52; --line: #2D3163;
  --ink: #F2F3FF; --muted: #9EA3CC; --violet: #6D4EF5; --violet-soft: #A796FF;
}
.stApp { background: var(--page); }
[data-testid="stHeader"] { background: var(--page); }
.block-container { padding-top: 4.25rem; padding-bottom: 2.5rem; max-width: 1440px; }
.msr {
  font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; font-size: 20px;
  line-height: 1; letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap;
  direction: ltr; font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased; vertical-align: middle;
}
.num { font-variant-numeric: tabular-nums; }

/* top bar */
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin-bottom: 6px; }
.brand { display: flex; align-items: center; gap: 14px; }
.logo {
  width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center; color: #fff; flex: none;
  background: linear-gradient(135deg, #3B3FE0, #8A4DFF 60%, #E8487E);
}
.logo .msr { font-size: 26px; }
.app-name { font-size: 1.35rem; font-weight: 600; color: var(--ink); line-height: 1.2; }
.greeting { color: var(--muted); font-size: 0.92rem; margin-top: 2px; }
.warning-line { display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 0.78rem; margin-top: 6px; }
.warning-line .msr { color: #BF7F1F; font-size: 16px; }

.section-title { font-size: 1.15rem; font-weight: 600; color: var(--ink); margin: 18px 0 12px; }
.section-title.tight { margin-bottom: 0; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 20px; padding: 18px; }

/* stat tiles */
.tiles { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.tile { background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 14px; min-width: 0; }
.tile-label { display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 0.78rem; white-space: nowrap; }
.tile-label i { width: 3px; height: 14px; border-radius: 2px; display: inline-block; }
.tile-value { font-size: 1.55rem; font-weight: 600; color: var(--ink); margin: 8px 0 2px; }
.tile-sub { font-size: 0.74rem; color: var(--muted); }
.tile-sub b { color: #3CC59A; font-weight: 600; }

/* panels built from Streamlit containers */
.st-key-check_panel, .st-key-model_panel {
  background: var(--panel); border: 1px solid var(--line); border-radius: 20px; padding: 18px;
}
.st-key-check_panel [data-testid="stBaseButton-secondary"] {
  background: var(--raised); border: 1px solid var(--line); color: var(--ink); font-size: 0.82rem; min-height: 2.3rem;
}
.st-key-check_panel [data-testid="stBaseButton-secondary"]:hover { border-color: var(--violet-soft); color: #fff; }
[data-testid="stBaseButton-primary"], [data-testid="stBaseLinkButton-primary"] { font-weight: 600; min-height: 2.75rem; }
.st-key-check_panel textarea { font-size: 0.95rem; }

/* results table */
.table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
.table th { text-align: left; color: var(--muted); font-weight: 500; font-size: 0.76rem; padding: 0 0 10px; }
.table td { padding: 10px 0; border-top: 1px solid var(--line); color: var(--ink); }
.table th:not(:first-child), .table td:not(:first-child) { text-align: right; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 9px; vertical-align: middle; }

/* prediction card: the one bold element, after the reference's bank card */
.card-wrap { position: relative; padding: 14px 18px 4px 0; }
.card-wrap::before {
  content: ""; position: absolute; inset: 0 0 18px 26px; border-radius: 22px; transform: rotate(5deg);
  background: linear-gradient(135deg, #E8487E, #F08A4B); opacity: 0.55;
}
.pred-card {
  position: relative; border-radius: 22px; padding: 22px 24px; min-height: 210px; color: #fff; overflow: hidden;
  display: flex; flex-direction: column; justify-content: space-between; gap: 14px;
  background:
    radial-gradient(circle at 88% 92%, rgba(52, 225, 163, 0.85) 0, rgba(52, 225, 163, 0) 42%),
    radial-gradient(circle at 72% 18%, rgba(255, 92, 168, 0.75) 0, rgba(255, 92, 168, 0) 38%),
    linear-gradient(135deg, #1F2BB8 0%, #4B35E8 50%, #8C4BFF 100%);
  box-shadow: 0 18px 40px rgba(20, 16, 80, 0.45);
}
.pc-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; }
.pc-caption { font-size: 0.82rem; opacity: 0.85; }
.pc-label { font-size: 2.3rem; font-weight: 700; line-height: 1.1; margin-top: 4px; }
.pc-label.waiting { font-size: 1.6rem; font-weight: 600; }
.pc-chip {
  display: flex; align-items: center; gap: 6px; padding: 6px 10px; border-radius: 999px; font-size: 0.82rem; font-weight: 600;
  background: rgba(255, 255, 255, 0.16); border: 1px solid rgba(255, 255, 255, 0.28); white-space: nowrap;
}
.pc-chip i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.7); }
.pc-text { font-size: 0.92rem; opacity: 0.92; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pc-text.wrap { white-space: normal; }
.pc-bottom { display: flex; justify-content: space-between; align-items: flex-end; gap: 12px; font-size: 0.8rem; line-height: 1.35; }
.pc-mark { font-weight: 700; font-style: italic; font-size: 1.15rem; letter-spacing: 0.5px; }

/* probability bars */
.bar-row { margin-bottom: 14px; }
.bar-row:last-child { margin-bottom: 2px; }
.bar-head { display: flex; justify-content: space-between; font-size: 0.86rem; color: var(--ink); margin-bottom: 6px; }
.bar-head.top { font-weight: 600; }
.bar-track { height: 8px; border-radius: 4px; background: var(--raised); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; }

/* moderation note and history items share one row layout */
.row { display: flex; align-items: center; gap: 12px; }
.row-icon { width: 38px; height: 38px; flex: none; border-radius: 12px; display: grid; place-items: center; color: #fff; }
.row-body { min-width: 0; flex: 1; }
.row-title { font-size: 0.9rem; font-weight: 600; color: var(--ink); }
.row-sub { font-size: 0.8rem; color: var(--muted); line-height: 1.4; }
.row-sub.clip { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.row-end { font-size: 0.82rem; color: var(--muted); white-space: nowrap; }
.history .row { padding: 10px 0; border-top: 1px solid var(--line); }
.history .row:first-child { border-top: none; padding-top: 0; }
.history .row:last-child { padding-bottom: 0; }

/* model profile and how-it-works */
.profile { display: flex; align-items: center; gap: 12px; background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 12px 14px; }
.avatar {
  width: 42px; height: 42px; border-radius: 50%; flex: none; display: grid; place-items: center; font-weight: 700;
  color: #fff; font-size: 0.85rem; background: linear-gradient(135deg, #1C9DBA, #6D4EF5);
}
.howto-badge { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.howto-badge .row-icon { background: var(--raised); color: var(--violet-soft); }
.howto-title { font-size: 1.05rem; font-weight: 600; color: var(--ink); line-height: 1.35; margin-bottom: 10px; }
.steps { margin: 0 0 12px; padding-left: 1.15rem; color: var(--muted); font-size: 0.84rem; line-height: 1.55; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 4px; }
.chip { display: flex; align-items: center; gap: 6px; font-size: 0.78rem; color: var(--ink); background: var(--raised); border-radius: 999px; padding: 5px 10px; }
.chip .msr { font-size: 16px; color: var(--violet-soft); }

@media (max-width: 640px) {
  .brand { align-items: flex-start; }
  .logo { width: 40px; height: 40px; border-radius: 12px; }
  .app-name { font-size: 1.2rem; }
  .tiles { grid-template-columns: 1fr; }
  .pc-label { font-size: 1.9rem; }
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


def moderation_note(conf):
    if conf < REVIEW_THRESHOLD:
        return f"Low confidence ({conf:.0%}): send to a human moderator for review."
    return f"Confidence {conf:.0%}. A human should still review any flagged post."


def excerpt(text, n):
    text = " ".join(text.split())
    return html.escape(text if len(text) <= n else text[: n - 1] + "…")


# ---------- state ----------
st.session_state.setdefault("text", "")
st.session_state.setdefault("history", [])
st.session_state.setdefault("last_result", None)

st.html(f"<style>{CSS}</style>")
tokenizer, model = load_model()

show(f"""
<div class="topbar">
  <div class="brand">
    <div class="logo">{icon("shield")}</div>
    <div>
      <div class="app-name">Swahili Hate &amp; Abuse Detector</div>
      <div class="greeting">Habari! Paste a Swahili post to check it for abusive or hateful language.</div>
      <div class="warning-line">{icon("warning")}<span>Content warning: shows hateful and abusive language. Built to assist
      human moderators, not to remove posts automatically.</span></div>
    </div>
  </div>
</div>
""")

left, middle, right = st.columns([1.3, 1.1, 0.95], gap="large")

# ---------- left: overview, input, test results ----------
with left:
    show(f"""
    <div class="section-title">Overview</div>
    <div class="tiles">
      <div class="tile">
        <div class="tile-label"><i style="background:#6D4EF5"></i>Macro-F1</div>
        <div class="tile-value num">0.899</div>
        <div class="tile-sub"><b>+0.023</b> over the TF-IDF baseline</div>
      </div>
      <div class="tile">
        <div class="tile-label"><i style="background:#1C9DBA"></i>Accuracy</div>
        <div class="tile-value num">90.7%</div>
        <div class="tile-sub">on 2,664 unseen test tweets</div>
      </div>
      <div class="tile">
        <div class="tile-label"><i style="background:#E8487E"></i>Hate recall</div>
        <div class="tile-value num">84.2%</div>
        <div class="tile-sub">of hateful test posts caught</div>
      </div>
    </div>
    <div class="section-title tight">Check a post</div>
    """)

    with st.container(key="check_panel"):
        st.caption("Try an example")
        for i, (col, (label, tip, text)) in enumerate(zip(st.columns(len(EXAMPLES)), EXAMPLES)):
            col.button(label, help=tip, on_click=use_example, args=(text,), width="stretch", key=f"example_{i}")
        st.text_area("Swahili post", key="text", height=130, placeholder="Andika chapisho hapa...")
        clicked = st.button("Classify post", type="primary", key="classify", width="stretch")

        if clicked:
            text = st.session_state.text
            if not text.strip():
                st.warning("Please enter some Swahili text.")
            else:
                probs = classify(text, tokenizer, model)
                label, conf = max(probs.items(), key=lambda kv: kv[1])
                result = {"text": text, "label": label, "conf": conf, "probs": probs, "note": moderation_note(conf)}
                st.session_state.last_result = result
                st.session_state.history = ([result] + st.session_state.history)[:HISTORY_SIZE]

    rows = "".join(
        f'<tr><td><span class="dot" style="background:{LABEL_COLOR[name]}"></span>{name}</td>'
        f'<td class="num">{p:.3f}</td><td class="num">{r:.3f}</td><td class="num">{f:.3f}</td></tr>'
        for name, (p, r, f) in TEST_RESULTS.items()
    )
    show(f"""
    <div class="section-title">Test-set results by class</div>
    <div class="panel"><table class="table">
      <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
      <tbody>{rows}</tbody>
    </table></div>
    """)

# ---------- middle: prediction card, probabilities, moderation note ----------
result = st.session_state.last_result
with middle:
    if result:
        color = LABEL_COLOR[result["label"]]
        card = f"""
        <div class="pc-top">
          <div><div class="pc-caption">Prediction</div><div class="pc-label">{result["label"]}</div></div>
          <div class="pc-chip"><i style="background:{color}"></i><span class="num">{result["conf"]:.1%}</span></div>
        </div>
        <div class="pc-text">“{excerpt(result["text"], 70)}”</div>
        <div class="pc-bottom"><span>{LABEL_HELP[result["label"]]}</span><span class="pc-mark">AfriHate</span></div>
        """
    else:
        card = """
        <div class="pc-top">
          <div><div class="pc-caption">Prediction</div><div class="pc-label waiting">Waiting for a post</div></div>
        </div>
        <div class="pc-text wrap">Pick an example or paste a post, then press Classify post.</div>
        <div class="pc-bottom"><span>Normal, Abuse or Hate</span><span class="pc-mark">AfriHate</span></div>
        """
    show(f'<div class="section-title">Result</div><div class="card-wrap"><div class="pred-card">{card}</div></div>')

    bars = ""
    for name in LABELS:
        p = result["probs"][name] if result else 0.0
        top = " top" if result and name == result["label"] else ""
        value = f"{p:.1%}" if result else "–"
        bars += f"""
        <div class="bar-row" title="{name}: {value}">
          <div class="bar-head{top}"><span>{name}</span><span class="num">{value}</span></div>
          <div class="bar-track"><div class="bar-fill" style="width:{p * 100:.1f}%;background:{LABEL_COLOR[name]}"></div></div>
        </div>"""
    show(f'<div class="section-title">Class probabilities</div><div class="panel">{bars}</div>')

    if result:
        low = result["conf"] < REVIEW_THRESHOLD
        note_icon, note_bg = ("flag", "#BF7F1F") if low else ("verified", "#6D4EF5")
        note_title = "Needs human review" if low else "Confident prediction"
        show(f"""
        <div class="section-title">Moderation note</div>
        <div class="panel"><div class="row">
          <div class="row-icon" style="background:{note_bg}">{icon(note_icon)}</div>
          <div class="row-body"><div class="row-title">{note_title}</div><div class="row-sub">{result["note"]}</div></div>
        </div></div>
        """)

# ---------- right: model profile, how it works, history ----------
with right:
    show(f"""
    <div class="section-title">Model</div>
    <div class="profile">
      <div class="avatar">AX</div>
      <div class="row-body"><div class="row-title">AfroXLMR-base</div>
      <div class="row-sub clip">{html.escape(MODEL_ID)}</div></div>
    </div>
    """)
    with st.container(key="model_panel"):
        show(f"""
        <div class="howto-badge"><div class="row-icon">{icon("psychology")}</div>
        <div class="row-sub">Fine-tuned on AfriHate (Swahili)</div></div>
        <div class="howto-title">How the model reads a post</div>
        <ol class="steps">
          <li>Splits the post into up to {MAX_LEN} subword tokens.</li>
          <li>A 12-layer Transformer turns them into contextual embeddings.</li>
          <li>A classification head and softmax give one probability per class.</li>
        </ol>
        <div class="chips">
          <span class="chip">{icon("memory")}278M parameters</span>
          <span class="chip">{icon("dataset")}13,876 training tweets</span>
        </div>
        """)
        st.link_button("View the model on Hugging Face", f"https://huggingface.co/{MODEL_ID}",
                       type="primary", width="stretch")

    items = ""
    for item in st.session_state.history:
        items += f"""
        <div class="row">
          <div class="row-icon" style="background:{LABEL_COLOR[item["label"]]}">{icon(LABEL_ICON[item["label"]])}</div>
          <div class="row-body"><div class="row-title">{item["label"]}</div>
          <div class="row-sub clip">{excerpt(item["text"], 60)}</div></div>
          <div class="row-end num">{item["conf"]:.0%}</div>
        </div>"""
    if not items:
        items = '<div class="row-sub">Posts you classify will show up here.</div>'
    show(f'<div class="section-title">History</div><div class="panel history">{items}</div>')
