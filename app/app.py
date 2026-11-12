import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------- Where the model files are ----------
ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

st.set_page_config(page_title="Support Ticket Router", page_icon="📨")


# ---------- Load everything once ----------
@st.cache_resource
def load():
    config = json.loads((MODELS / "config.json").read_text())
    encoder = SentenceTransformer(config["encoder"])
    clf = joblib.load(MODELS / "intent_classifier.joblib")
    intent_to_dept = {
        i: d for d, intents in config["departments"].items() for i in intents
    }
    dept_of_class = np.array([intent_to_dept[c] for c in clf.classes_])
    return config, encoder, clf, dept_of_class


config, encoder, clf, dept_of_class = load()


# ---------- The routing logic ----------
def route(message):
    p = clf.predict_proba(encoder.encode([message], normalize_embeddings=True))[0]
    dept_scores = {d: float(p[dept_of_class == d].sum()) for d in config["departments"]}
    dept = max(dept_scores, key=dept_scores.get)
    top3 = np.argsort(p)[::-1][:3]
    top_intents = [(clf.classes_[i], float(p[i])) for i in top3]
    return dept, dept_scores[dept], dept_scores, top_intents


# ---------- The page ----------
st.title("📨 Support Ticket Router")
st.caption(
    "Reads a customer support message and sends it to the right department. "
    "If the model isn't confident enough, the message goes to a human agent. "
    "Trained on 13,000 banking support messages (BANKING77 dataset)."
)

examples = [
    "my card still hasnt arrived its been 2 weeks",
    "someone used my card in another country, i didnt do that",
    "why was I charged a fee when I took cash out?",
    "how do I send money to my brother's account?",
    "my top up didnt go through",
    "I need to change my address",
    "hello",
]
choice = st.selectbox("Pick an example, or write your own below", [""] + examples)
message = st.text_area(
    "Customer message",
    value=choice,
    height=100,
    placeholder="e.g. My card hasn't arrived yet",
)

if st.button("Route message", type="primary") and message.strip():
    dept, conf, dept_scores, top_intents = route(message)

    if conf >= config["threshold"]:
        st.success(f"### ➡️ Route to: {dept}")
    else:
        st.warning(
            "### 🙋 Route to: Human agent\n"
            "The model isn't confident enough to route this automatically."
        )

    st.metric(
        "Confidence",
        f"{conf:.0%}",
        help=f"Messages below {config['threshold']:.0%} go to a human agent.",
    )

    st.write("**Most likely issues:**")
    for intent, prob in top_intents:
        st.write(f"- {intent.replace('_', ' ').rstrip('?').capitalize()}: {prob:.0%}")

    st.write("**Confidence by department:**")
    st.bar_chart(pd.Series(dept_scores).sort_values(), horizontal=True)
