import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)

# Model load karo
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬", layout="wide")
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("NLP Project | **Virajbhai Mavani**")
st.divider()

# ── PREDICTION SECTION ──
movie_name = st.text_input("🎬 Enter Movie Name:", placeholder="e.g. Inception, The Dark Knight...")
review = st.text_area("✍️ Enter Movie Review:", height=150, placeholder="Type your movie review here...")

if st.button("🔍 Analyze Sentiment", use_container_width=True):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review!")
    else:
        cleaned = clean_text(review)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        movie_display = movie_name.strip() if movie_name.strip() != "" else "Not specified"

        if prediction == 'positive':
            st.success(f"### 😊 Positive Review!")
        else:
            st.error(f"### 😞 Negative Review!")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Confidence", f"{max(probability)*100:.1f}%")
        with col_b:
            st.metric("Movie", movie_display)
        
        sentiment_label = "Positive 😊" if prediction == 'positive' else "Negative 😞"
        st.info(f"🎬 **Movie:** {movie_display}   |   📊 **Review Sentiment:** {sentiment_label}   |   🎯 **Confidence:** {max(probability)*100:.1f}%")

st.divider()

# ── KEY FINDINGS ──
st.markdown("### 📊 Key Findings from Analysis")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Reviews", "49,582")
with col2:
    st.metric("Best Accuracy", "89.00%")
with col3:
    st.metric("Best Model", "Logistic Regression")

st.divider()

# ── EDA CHARTS ──
st.markdown("### 📈 Data Insights")

col1, col2 = st.columns(2)

# Chart 1: Sentiment Distribution
with col1:
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    df_viz = pd.DataFrame({'sentiment': ['Positive', 'Negative'],
                           'count': [24884, 24698]})
    ax1.bar(df_viz['sentiment'], df_viz['count'],
            color=['#2ecc71', '#e74c3c'])
    ax1.set_title('Sentiment Distribution')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Sentiment')
    for i, v in enumerate(df_viz['count']):
        ax1.text(i, v + 50, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig1)

# Chart 2: Average Review Length
with col2:
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sentiments = ['Negative', 'Positive']
    lengths = [1270, 1303]
    ax2.bar(sentiments, lengths, color=['#e74c3c', '#2ecc71'])
    ax2.set_title('Average Review Length by Sentiment')
    ax2.set_ylabel('Average Length (characters)')
    ax2.set_xlabel('Sentiment')
    for i, v in enumerate(lengths):
        ax2.text(i, v + 5, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig2)

col3, col4 = st.columns(2)

# Chart 3: Model Comparison
with col3:
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    models = ['Logistic\nRegression', 'Naive\nBayes', 'Random\nForest']
    accuracies = [89.00, 85.34, 83.77]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    ax3.bar(models, accuracies, color=colors)
    ax3.set_title('Model Comparison')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_ylim(80, 92)
    for i, v in enumerate(accuracies):
        ax3.text(i, v + 0.1, f'{v}%', ha='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig3)

# Chart 4: Top 10 Most Common Words
with col4:
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    words = ['movie', 'film', 'great', 'good', 'story',
             'love', 'character', 'best', 'time', 'watch']
    counts = [45000, 42000, 38000, 35000, 32000,
              30000, 28000, 26000, 24000, 22000]
    ax4.barh(words[::-1], counts[::-1], color='#3498db')
    ax4.set_title('Top 10 Most Common Words')
    ax4.set_xlabel('Frequency')
    plt.tight_layout()
    st.pyplot(fig4)

st.divider()

# ── MODEL PERFORMANCE ──
st.markdown("### 🏆 Model Performance")
col5, col6, col7 = st.columns(3)
with col5:
    st.metric("Logistic Regression", "89.00%", "🥇 Best Model")
with col6:
    st.metric("Naive Bayes", "85.34%", "🥈")
with col7:
    st.metric("Random Forest", "83.77%", "🥉")