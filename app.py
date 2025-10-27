import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Hello, Dale", layout="centered")
st.title("🧠 Tiny Sentiment Analyzer")
st.caption("Runs locally on CPU. No data is stored.")

@st.cache_resource(show_spinner=False)
def load_model():
    # Small, fast model that works on Streamlit Cloud
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

clf = load_model()

text = st.text_area(
    "Paste some text to analyze:",
    height=160,
    placeholder="e.g., I love this! It’s fast and easy to use."
)
# comment 
if st.button("Analyze sentiment"):
    if not text.strip():
        st.warning("Please paste some text first.")
    else:
        with st.spinner("Thinking..."):
            result = clf(text)[0]
        label = result["label"].title()
        score = float(result["score"])
        st.success(f"**Sentiment:** {label} — confidence: {score:.3f}")
        if label.upper() == "POSITIVE":
            st.balloons()

st.divider()
st.write("Try examples:")
cols = st.columns(3)
examples = [
    "I absolutely love this product. Great job!",
    "This is terrible and I’m very disappointed.",
    "It’s okay—some good parts, some bad."
]
for c, ex in zip(cols, examples):
    if c.button(ex):
        st.session_state["example"] = ex

if "example" in st.session_state:
    st.info(f"Example selected: {st.session_state['example']}")
