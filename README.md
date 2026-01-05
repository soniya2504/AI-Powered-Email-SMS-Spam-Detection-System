# AI-Powered Email & SMS Spam Detection System

## Project Overview
This project implements an end-to-end Natural Language Processing (NLP) and Machine Learning pipeline to classify email and SMS messages as **Spam** or **Not Spam**. The solution uses **TF-IDF vectorization** and a **Linear Support Vector Machine (LinearSVC)** classifier, supported by robust text preprocessing, model benchmarking, error analysis, and deployment-ready model serialization.

The project is part of a **Data Science & NLP portfolio**, demonstrating practical skills in text cleaning, feature engineering, supervised learning, and lightweight application deployment.

---

## Key Features
- **Comprehensive Text Preprocessing** using regex and token normalization
- **TF-IDF feature extraction** for high-dimensional sparse text representation
- **Benchmarking 3 ML models**: Logistic Regression, Naive Bayes, Linear SVM
- **Stratified Train-Test Split** to preserve class distribution
- **Misclassification Analysis** to inspect model failure cases
- **Serialized pipeline export** using `joblib` for deployment
- **Recruiter-ready UI deployment** support via Streamlit (optional integration)

---

## Tech Stack
- **Programming:** Python
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `nltk`, `joblib`, `regex`
- **NLP Methods:** Tokenization, stop-word removal, stemming, TF-IDF vectorization
- **ML Model:** LinearSVC (final production pipeline)

---

## Dataset
- Source contains labeled Email/SMS text with **ham** and **spam** classes
- After cleaning and normalization:
  - **Total samples:** 5090
  - **Spam:** 594
  - **Not Spam:** 4496
  - **Columns:** Raw Text | Cleaned Text | Label

---

## Model Performance
| Model | Accuracy |
|--------|----------|
| Logistic Regression | 96.07% |
| Naive Bayes | 95.77% |
| Linear SVM | **97.54%** |

- **Final Test Accuracy (SVM pipeline):** 97.25%
- Strong precision on both Spam and Not Spam classes
- Misclassified samples inspected and logged for transparency

---

## Project Structure
