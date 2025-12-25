import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("Analysis Dashboard")

# Load results
results = pd.read_csv("data/sentiment_results.csv")

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
