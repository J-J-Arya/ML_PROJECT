# SUMMARIZER USING TextRank (with summa)

import streamlit as st
from summa import summarizer  # ✅ Works like gensim.summarization

st.set_page_config(page_title="📰 News Summarizer", layout="centered")

st.title("📰 News Article Summarizer")
st.write("Summarize long news articles using the **TextRank** algorithm (via summa).")

# ---------- User Input ----------
text_input = st.text_area("✏️ Paste your news article here", height=300)

summary_ratio = st.slider("🧠 Summary Ratio (10% - 90%)", min_value=0.1, max_value=0.9, step=0.1, value=0.3)

# ---------- Summarization Logic ----------
if st.button("Summarize"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter a news article to summarize.")
    else:
        try:
            summary = summarizer.summarize(text_input, ratio=summary_ratio)
            st.subheader("✅ Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")