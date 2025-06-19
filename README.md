# 🩺 Diabetes Prediction Model

This project uses machine learning to predict whether a patient is likely to have diabetes, based on diagnostic health indicators. The application is built with Python and deployed via a **Streamlit** web interface for interactive and real-time predictions.

## 📌 Project Overview

- **Model**: Logistic Regression (can be extended to other ML models)
- **Frameworks**: Streamlit, Scikit-learn, Pandas, NumPy
- **Dataset**: Pima Indians Diabetes Dataset (from Kaggle)
- **Features Used**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age

## 💡 Features

- **Interactive UI**: Enter patient parameters through the web app.
- **Batch and Real-Time Prediction Ready**: Model accepts individual or multiple entries.
- **Clean Code & Reproducible Pipeline**: Includes preprocessing, training, and model persistence.
- **Accurate Prediction**: Optimized logistic regression model trained on a clean dataset.

## 🧠 Machine Learning Pipeline

1. Data Cleaning & Preprocessing
2. Feature Scaling
3. Model Training (Logistic Regression)
4. Evaluation using accuracy, confusion matrix
5. Deployment using Streamlit
