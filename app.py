import streamlit as st
import joblib, re
from pathlib import Path


st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# ---------- PATHS ----------
# Fix the __file__ name (double underscores)
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "spam_classifier.joblib"
VECTORIZER_PATH = ROOT / "models" / "tfidf.joblib"

# ---------- LOAD ARTIFACTS ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_artifacts()

# ---------- CLEANING ----------
def clean_text_simple(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- PAGE STYLE ----------
st.markdown("""
<style>
body { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%); }
.card { padding: 1.2rem; border-radius: 12px; box-shadow: 0 6px 18px rgba(15,23,42,0.06); }
header { display:none; }
.footer { color: #6b7280; font-size:12px; margin-top: 12px; }
</style>
""", unsafe_allow_html=True)

# ---------- UI ----------
st.title("📧 Spam Detector")
st.subheader("Paste an email or SMS below and press Predict — simple, fast, and private.")

with st.container():
    user_input = st.text_area("Message text", height=200, placeholder="Type or paste the message to classify...")

    if st.button("Predict"):
        if not user_input.strip():
            st.error("Please enter some text to classify.")
        else:
            cleaned = clean_text_simple(user_input)
            X = vectorizer.transform([cleaned])
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0]
            label = "Spam" if pred == 1 else "Ham (not spam)"
            confidence = max(prob)

            if pred == 1:
                st.error(f"🚨 **Prediction:** {label}\n\nConfidence: `{confidence:.2f}`")
            else:
                st.success(f"✅ **Prediction:** {label}\n\nConfidence: `{confidence:.2f}`")

st.markdown('<p class="footer">Built locally using Streamlit 🧠</p>', unsafe_allow_html=True)
