import streamlit as st
from transformers import pipeline

# --- Page setup ---
st.set_page_config(page_title="Welcome!", layout="centered")
st.title("🧠 Sentiment Analyzer")
st.caption("Runs locally on CPU. No data is stored.")

# --- Load the model (cached so it only loads once) ---
@st.cache_resource(show_spinner=False)
def load_model():
    # This model outputs three labels: negative, neutral, positive
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

clf = load_model()

# --- User text input ---
text = st.text_area(
    "Paste some text to analyze:",
    height=160,
    placeholder="e.g., I love this! It’s fast and easy to use."
)

# --- Sentiment analysis button ---
if st.button("Analyze sentiment"):
    if not text.strip():
        st.warning("Please paste some text first.")
    else:
        with st.spinner("Thinking..."):
            results = clf(text)
        
        # The model returns a list of dictionaries (one per input)
        result = results[0]
        label = result["label"].replace("LABEL_", "")
        score = float(result["score"])

        # --- Interpretation ---
        # Convert label IDs (0,1,2) into text descriptions
        label_map = {"0": "Negative", "1": "Neutral", "2": "Positive"}
        sentiment = label_map.get(label, label.title())

        # --- Display result ---
        if sentiment == "Positive":
            st.success(f"**Sentiment:** 😊 {sentiment} — confidence: {score:.3f}")
            st.balloons()
        elif sentiment == "Negative":
            st.error(f"**Sentiment:** 😠 {sentiment} — confidence: {score:.3f}")
        else:
            st.info(f"**Sentiment:** 😐 {sentiment} — confidence: {score:.3f}")

# --- Example section ---
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
