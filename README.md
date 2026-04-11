# 🎬 Movie Review Sentiment Analyzer

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F3C?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

A Natural Language Processing (NLP) project that analyzes movie reviews and
predicts whether the sentiment is positive or negative using Machine Learning.
The goal is to build an accurate text classification model and deploy it as
an interactive web application.

---

## 🌐 Live Demo

Check out the interactive Streamlit app:

**[Launch App](https://movie-sentiment-analyzer-rrxkbqimw8njc2eqbjjlxu.streamlit.app/)**

---

## 📊 Table of Contents

- [Project Overview](#-project-overview)
- [Live Demo](#-live-demo)
- [Dataset](#-dataset)
- [Tools & Libraries](#️-tools--libraries)
- [Project Structure](#-project-structure)
- [What I Did](#-what-i-did)
- [Key Findings](#-key-findings)
- [Model Performance](#-model-performance)
- [How to Run Locally](#️-how-to-run-locally)

---

## 📖 Project Overview

This project performs end-to-end NLP analysis on the IMDB Movie Reviews dataset.
It covers data cleaning, SQL-based analysis using SQLite, exploratory data analysis,
text preprocessing with NLTK, TF-IDF feature engineering, training multiple ML models,
and a live Streamlit web app where users can enter any movie review and get
instant sentiment prediction with confidence score.

---

## 📦 Dataset

- **Source:** IMDB Dataset of 50K Movie Reviews (Kaggle)
- **Original Authors:** Maas et al., ACL 2011
- **Records:** 49,582 reviews (after cleaning)
- **Domain:** Movie Reviews — Binary Sentiment Classification
- **Features:** review (text), sentiment (positive/negative)
- **Link:** http://ai.stanford.edu/~amaas/data/sentiment/

> **Note:** IMDB Dataset.csv (63MB) is not included due to GitHub size limits.
> Download from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## 🛠️ Tools & Libraries

| Tool / Library | Purpose |
|----------------|---------|
| Python 3 | Core programming language |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualizations |
| Seaborn | Statistical visualizations |
| NLTK | Text preprocessing and stopwords |
| Scikit-learn | ML models and TF-IDF vectorization |
| SQLite | SQL-based data storage and analysis |
| WordCloud | Word frequency visualization |
| Streamlit | Interactive web app |
| Jupyter Notebook | Analysis and exploration |

---

## 📂 Project Structure

```
movie-sentiment-analyzer/
│
├── App.py                        # Streamlit web application
├── sentiment_analysis.ipynb      # Main analysis and model training notebook
├── model.pkl                     # Trained Logistic Regression model
├── tfidf.pkl                     # Fitted TF-IDF vectorizer
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## ✅ What I Did

- 🔹 Performed Data Cleaning — removed HTML tags, duplicates and stopwords using NLTK
- 🔹 Stored data in SQLite database and ran SQL queries for sentiment insights
- 🔹 Built Word Cloud visualizations for positive and negative reviews
- 🔹 Performed EDA using Matplotlib and Seaborn — review length, sentiment distribution
- 🔹 Applied TF-IDF vectorization with 5,000 features
- 🔹 Compared 3 ML models — Logistic Regression, Naive Bayes, Random Forest
- 🔹 Deployed Streamlit app with movie name input, confidence score and data insights

---

## 🔍 Key Findings

1. **Balanced dataset** — 24,884 positive and 24,698 negative reviews
2. **Positive reviews are longer** — avg 1,303 characters vs 1,270 for negative
3. **Most common words** — movie, film, great, good, story, love, character
4. **Stopword removal** significantly improved model performance
5. **Logistic Regression** outperformed Naive Bayes and Random Forest for NLP
6. **TF-IDF** with 5,000 features gave best balance of speed and accuracy

---

## 🤖 Model Performance

| Model | Accuracy |
|-------|----------|
| **Logistic Regression** ✅ | **89.00%** |
| Naive Bayes | 85.34% |
| Random Forest | 83.77% |

**Best Model: Logistic Regression — 89% accuracy**

---

## ⚙️ How to Run Locally

1. Clone this repository
   ```bash
   git clone https://github.com/viraj01-coder/movie-sentiment-analyzer.git
   cd movie-sentiment-analyzer
   ```

2. Install required libraries
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and place it in the root directory as:
   ```
   IMDB Dataset.csv
   ```

4. Run the Streamlit app
   ```bash
   streamlit run App.py
   ```

5. Open the notebook
   ```bash
   jupyter notebook sentiment_analysis.ipynb
   ```

---

*Dataset: IMDB Movie Reviews — Maas et al., ACL 2011 | Author: Virajbhai Mavani*
