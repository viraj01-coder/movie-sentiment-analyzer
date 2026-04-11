# Movie Review Sentiment Analyzer

A Natural Language Processing (NLP) project that analyzes movie reviews and
predicts whether the sentiment is positive or negative using Machine Learning.
The goal is to build an accurate text classification model and deploy it as
an interactive web application.

---

## Live Demo

Check out the interactive Streamlit app:

**[Launch App](https://movie-sentiment-analyzer-rrxkbqimw8njc2eqbjjlxu.streamlit.app/)**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Live Demo](#live-demo)
- [Dataset](#dataset)
- [Tools & Libraries](#tools--libraries)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [How to Run Locally](#how-to-run-locally)

---

## Project Overview

This project performs end-to-end NLP analysis on the IMDB Movie Reviews dataset.
It covers data cleaning, SQL-based analysis using SQLite, exploratory data analysis,
text preprocessing with NLTK, TF-IDF feature engineering, training multiple ML models,
and a live Streamlit web app where users can enter any movie review and get
instant sentiment prediction with confidence score.

---

## Dataset

- **Source:** IMDB Dataset of 50K Movie Reviews (Kaggle)
- **Original Authors:** Maas et al., ACL 2011
- **Records:** 49,582 reviews (after cleaning)
- **Domain:** Movie Reviews — Binary Sentiment Classification
- **Features:** review (text), sentiment (positive/negative)
- **Link:** http://ai.stanford.edu/~amaas/data/sentiment/

---

## Tools & Libraries

| Tool / Library | Purpose |
|----------------|---------|
| Python 3 | Core programming language |
| Pandas | Data manipulation and analysis |
| NLTK | Text preprocessing and stopwords |
| Scikit-learn | ML models and TF-IDF vectorization |
| SQLite | SQL-based data storage and analysis |
| Matplotlib | Data visualizations |
| WordCloud | Word frequency visualization |
| Streamlit | Interactive web app |
| Jupyter Notebook | Analysis and exploration |

---

## Project Structure

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

> **Note:** IMDB Dataset.csv (63MB) is not included due to GitHub size limits.
> Download from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## Key Findings

1. **Balanced dataset** — 24,884 positive and 24,698 negative reviews
2. **Positive reviews are longer** — avg 1,303 characters vs 1,270 for negative
3. **Most common words** — movie, film, great, good, story, love, character
4. **Stopword removal** significantly improved model performance
5. **Logistic Regression** outperformed Naive Bayes and Random Forest for NLP
6. **TF-IDF** with 5,000 features gave best balance of speed and accuracy

---

## Model Performance

| Model | Accuracy |
|-------|----------|
| **Logistic Regression** ✅ | **89.00%** |
| Naive Bayes | 85.34% |
| Random Forest | 83.77% |

**Best Model: Logistic Regression — 89% accuracy**

---

## How to Run Locally

1. Clone this repository
   ```
   git clone https://github.com/viraj01-coder/movie-sentiment-analyzer.git
   ```

2. Install required libraries
   ```
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and place it in the root directory as:
   ```
   IMDB Dataset.csv
   ```

4. Run the Streamlit app
   ```
   streamlit run App.py
   ```

5. Open the notebook
   ```
   jupyter notebook sentiment_analysis.ipynb
   ```

---

*Dataset: IMDB Movie Reviews — Maas et al., ACL 2011 | Author: Virajbhai Mavani*
