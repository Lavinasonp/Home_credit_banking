# 🏦 Bank Credit Risk Assessment System

**End-to-end Bank Credit Risk Assessment System using LightGBM to predict loan default probability and support data-driven lending decisions.**

---

## 📌 Executive Summary

Banks face a critical challenge in balancing loan approvals with default risk.  
This project presents a machine learning–based **Credit Risk Assessment System** that predicts the probability of loan default using applicant demographic, financial, and credit history data.

The system is designed to handle **highly imbalanced real-world banking data**, apply business-aware evaluation metrics, and provide insights that help financial institutions make **data-driven and risk-sensitive lending decisions**.

---

## 📊 Dataset Risk Overview

![Target Distribution](images/target_distribution.png)

- The dataset is highly imbalanced, with only **~8% default cases**, reflecting real-world banking scenarios  
- Accuracy alone is insufficient for credit risk modeling due to asymmetric costs  
- This motivates the use of **ROC-AUC, recall-focused metrics, and threshold optimization**

---

## 📌 Overview

Financial institutions face significant losses due to loan defaults.  
This project addresses the problem by building a machine learning system that predicts **customer credit risk** using historical banking data, enabling **proactive and data-backed credit approval decisions**.

The solution is designed with **real-world banking constraints**, focusing not only on model accuracy but also on **risk-sensitive decision making**.

---

## 🧠 Solution Architecture

The system follows a modular and scalable machine learning pipeline:

- Data ingestion from multiple banking-related sources  
- Exploratory Data Analysis (EDA) to understand risk patterns  
- Feature engineering and preprocessing  
- Model training using stratified cross-validation  
- Risk probability prediction for decision support  

---

## 🤖 Machine Learning Approach

- Implemented **LightGBM Classifier** optimized for imbalanced credit risk data  
- Used **Stratified K-Fold Cross-Validation** to ensure unbiased and stable performance  
- Evaluated models using:
  - ROC-AUC
  - Confusion Matrix
  - Classification Report  
- Applied **business-driven probability thresholding** instead of default cut-offs to reduce high-risk false approvals  

> This approach aligns model predictions with real-world banking risk tolerance.

---

## 📊 Data & Feature Engineering

- Integrated multiple banking data sources including application and bureau-related information  
- Handled missing values and encoded categorical variables  
- Scaled numerical features for model stability  
- Performed EDA to analyze feature distributions and credit risk trends  

---

## 📈 Results & Evaluation

- Achieved **stable ROC-AUC performance** across cross-validation folds  
- Optimized decision threshold to **minimize false negatives**, reducing potential default risk  
- Final model balances **predictive performance and business impact**

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries & Tools:**  
  - Pandas  
  - NumPy  
  - Scikit-learn  
  - LightGBM  
  - Matplotlib  
  - Seaborn  
  - Joblib  

---

## 📁 Project Structure

```
├── eda.ipynb              # Exploratory Data Analysis
├── model_training.py     # Model training and evaluation pipeline
├── main.py               # End-to-end execution script
├── requirements.txt      # Project dependencies
```

---

## ▶️ How to Run the Project

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd bank-credit-risk-system
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the pipeline
   ```bash
   python main.py
   ```

---

## 🚀 Skills Demonstrated

- Applied Machine Learning for financial risk modeling  
- Handling imbalanced datasets and business-driven evaluation  
- End-to-end ML pipeline development  
- Feature engineering and exploratory data analysis  
- Model evaluation aligned with real-world decision constraints  
- Clean, modular, and reusable ML code design  

---

## 📌 Project Status

This project follows **industry-level ML engineering practices** and is structured for:
- Scalability  
- Experimentation  
- Reproducibility  

Future improvements may include model explainability, hyperparameter optimization, and deployment.

---

### ⭐ Why This Project Stands Out

- Solves a real-world banking risk problem  
- Uses industry-relevant machine learning techniques  
- Balances technical performance with business impact  
- Follows professional ML engineering standards  
