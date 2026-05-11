import streamlit as st
import pandas as pd
from transformers import pipeline

# Load model once and cache it
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

sentiment_pipeline = load_model()

# Convert sentiment score to NPS category
def score_to_nps(score):
    if score > 0.5:
        return "Promoter"
    elif score < -0.5:
        return "Detractor"
    else:
        return "Neutral"

st.title("📊 NPS Text Analyzer")
st.markdown("Analyze customer feedback sentiment and calculate NPS score.")

# Two modes: upload CSV or enter text manually
mode = st.radio("Choose input mode:", ("Upload CSV file", "Enter text manually"))

feedbacks = []

if mode == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload a CSV file with a 'feedback' column", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "feedback" in df.columns:
                feedbacks = df["feedback"].astype(str).tolist()
            else:
                st.error("CSV must contain a 'feedback' column.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
else:
    # Manual input (one line per feedback)
    text_input = st.text_area("Enter customer feedback (one per line)", height=200)
    if text_input:
        feedbacks = [line.strip() for line in text_input.split("\n") if line.strip()]

if st.button("Analyze") and feedbacks:
    results = []
    for fb in feedbacks:
        try:
            result = sentiment_pipeline(fb)[0]
            label = result["label"]
            conf = result["score"]
            signed_conf = conf if label == "POSITIVE" else -conf
            nps_cat = score_to_nps(signed_conf)
            results.append({
                "Feedback": fb,
                "Sentiment": label,
                "Confidence": f"{conf:.3f}",
                "NPS Category": nps_cat
            })
        except Exception:
            results.append({
                "Feedback": fb,
                "Sentiment": "Error",
                "Confidence": "N/A",
                "NPS Category": "Error"
            })

    if results:
        df_res = pd.DataFrame(results)
        st.subheader("🔍 Analysis Results")
        st.dataframe(df_res, use_container_width=True)

        # NPS calculation
        promoters = sum(1 for r in results if r["NPS Category"] == "Promoter")
        detractors = sum(1 for r in results if r["NPS Category"] == "Detractor")
        total = len(results)
        if total > 0:
            nps_score = (promoters - detractors) / total * 100
            st.subheader("📊 NPS Statistics")
            st.metric("NPS Score", f"{nps_score:.1f}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Promoters", promoters)
            with col2:
                st.metric("Detractors", detractors)
            with col3:
                st.metric("Neutrals", total - promoters - detractors)