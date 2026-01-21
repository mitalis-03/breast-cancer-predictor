# 🩺 Breast Cancer Diagnostic System

A machine learning-powered web application that classifies breast tumors as **Malignant** or **Benign** based on clinical biopsy measurements. 

**Live Demo:** https://breast-cancer-predictor-cahmbouighed9b25bqbnek.streamlit.app/
---

## 📌 Project Overview
Breast cancer is one of the most common cancers globally. Early and accurate diagnosis is critical for effective treatment. This project utilizes the **Wisconsin Diagnostic Breast Cancer (WDBC) Dataset** to train a Random Forest model capable of predicting tumor types with high precision.

### Key Features
- **Data-Driven Insights:** Uses 30 clinical features including nucleus texture, area, and smoothness.
- **Optimized Performance:** Implements feature selection to remove multi-collinearity (redundant features).
- **Interactive Dashboard:** A Streamlit-based web app where medical professionals can input data via sliders and get instant results.
- **Robust Validation:** Validated using 5-Fold Cross-Validation to ensure model stability across different patient groups.

---

## 🛠️ Technologies Used
- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib
- **Model:** Random Forest Classifier
- **Deployment:** Streamlit & Streamlit Cloud
- **Persistence:** Joblib (for model/scaler export)

---

## 🧬 Machine Learning Pipeline

### 1. Data Cleaning & Pre-processing
- Handled missing values and verified data types.
- Encoded target variables (0: Malignant, 1: Benign).
- Applied **StandardScaler** to normalize feature ranges for better model convergence.

### 2. Feature Engineering
- Analyzed correlation matrices to identify highly correlated features (e.g., radius, perimeter, and area).
- Dropped redundant features with a correlation threshold > 0.95 to prevent overfitting.

### 3. Model Training
- Algorithm: **Random Forest Classifier**
- Hyperparameters: `n_estimators=100`, `random_state=42`.
- Accuracy achieved: ~98% (Initial Test Set).

### 4. Evaluation
- **Confusion Matrix:** Specifically monitored for "False Negatives" to ensure high recall for malignant cases.
- **Cross-Validation:** Confirmed model consistency with a 5-fold CV score.

---
