import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
# Title
st.title("Analysis Dashboard")

# Load results
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "sentiment_results.csv"
results = pd.read_csv(DATA_PATH)
# Show raw data
st.subheader("Raw Data")
st.dataframe(results)

# Show statistics
st.subheader("Sentiment Distribution")
st.write(results['predicted_sentiment'].value_counts())

# Plot a countplot using seaborn
fig, ax = plt.subplots()
sns.countplot(x='predicted_sentiment', data=results, ax=ax)
st.pyplot(fig)

# Optional: Filter by sentiment
st.subheader("Filter by Sentiment")
option = st.selectbox("Choose sentiment", results['predicted_sentiment'].unique())
filtered = results[results['predicted_sentiment'] == option]
st.dataframe(filtered)

import json

SUMMARY_PATH = BASE_DIR / "data" / "summaries.json"

with open(SUMMARY_PATH) as f:
    summaries = json.load(f)

emotion_summaries = summaries["emotion_summaries"]
# ----------------------
# Overall summary
# ----------------------
st.subheader("🧾 Overall Summary")
st.info(summaries["overall_summary"])

# ----------------------
# Sentiment-wise summary
# ----------------------
st.subheader("📝 Sentiment-wise Summary")

selected_sentiment = st.selectbox(
    "Select sentiment to view summary",
    options=list(emotion_summaries.keys())
)

st.success(emotion_summaries[selected_sentiment])
