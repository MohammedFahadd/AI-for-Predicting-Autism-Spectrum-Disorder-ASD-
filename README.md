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

<div align="center">
  <img src="screenshots/eda1.png" width="400"/>
  <img src="screenshots/eda2.png" width="400"/>
  <img src="screenshots/eda3.png" width="400"/>
  <img src="screenshots/eda4.png" width="300"/>
</div>

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
