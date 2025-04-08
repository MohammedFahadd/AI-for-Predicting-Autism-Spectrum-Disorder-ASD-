# AI-for-Predicting-Autism-Spectrum-Disorder-ASD-
AI for Predicting Autism Spectrum Disorder (ASD) This repository contains an end-to-end machine learning project that predicts the likelihood of Autism Spectrum Disorder (ASD) based on user screening responses and demographic data.

# 🧠 AI for Predicting Autism Spectrum Disorder (ASD)

This project utilizes machine learning models to predict Autism Spectrum Disorder (ASD) based on clinical and behavioral screening data. By analyzing patterns in age, gender, screening scores, and family/social factors, the system helps in identifying early indicators of ASD.

## 📊 Exploratory Data Analysis (EDA)

We performed detailed EDA to understand correlations between features and ASD likelihood. Some key visual insights:

- **Screening Questions**: Clear separation observed between ASD (1) and non-ASD (0) classes across binary behavioral features.
- **Gender & Age**: Slightly higher ASD detection among males; age distribution shows younger individuals are more likely in ASD class.
- **Ethnicity & Nationality**: Class distribution varies, highlighting demographic influence.
- **Screening Results**: Higher screening scores correspond to ASD-positive predictions.
- **Pie Chart Insight**: Dataset is imbalanced (76.9% non-ASD vs 23.1% ASD).

## 🧬 Dataset

We use ASD screening data from publicly available sources, including:
- Age
- Gender
- Screening answers (Q1–Q10)
- Family history
- Communication & social behavior patterns
- Ethnicity, nationality, and relation to individual

## 🧠 Machine Learning Models

Trained and evaluated:
- Logistic Regression
- Random Forest
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- (Optional) Deep Learning models using Keras

Model performance is evaluated using:
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC curve

## 📂 Project Structure
ASD-Prediction/
├── data/
├── notebooks/
├── models/
├── outputs/ (for plots)
├── app/ (if applicable)
├── README.md
└── requirements.txt

## 🚀 How to Run

bash
# Clone the repo
git clone https://github.com/your-username/asd-prediction.git
cd asd-prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/asd_analysis.ipynb
