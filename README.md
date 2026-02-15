# ML_Assignment_2
repo for ML assignment
# ML_Assignment_2
repo for ML assignment
# ML Assignment 2 - Classification Models Dashboard

## Problem Statement
The goal of this project is to implement and compare multiple machine learning classification models on a real-world dataset and deploy them using a Streamlit web application.

The application allows users to upload a dataset, select a classification model, and view evaluation metrics and confusion matrix.

---

## Dataset Description

Dataset Name: Adult Income Dataset

Source: UCI Machine Learning Repository

Number of Instances: 48,842

Number of Features: 14

Target Variable: income

Classes:
- <=50K
- >50K

Objective:
Predict whether a person's income exceeds $50K per year based on demographic and employment features.

---

## Models Used and Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------|---------|------|-----------|--------|----------|------|
| Logistic Regression | 0.8247 | 0.8568 | 0.7113 | 0.4596 | 0.5584 | 0.4722 |
| Decision Tree | 0.8128 | 0.7516 | 0.6074 | 0.6334 | 0.6201 | 0.4962 |
| KNN | 0.8353 | 0.8572 | 0.6786 | 0.6022 | 0.6381 | 0.5335 |
| Naive Bayes | 0.8093 | 0.8613 | 0.7064 | 0.3584 | 0.4755 | 0.4060 |
| Random Forest | 0.8587 | 0.9091 | 0.7427 | 0.6340 | 0.6841 | 0.5969 |
| XGBoost | 0.8719 | 0.9270 | 0.7661 | 0.6754 | 0.7179 | 0.6376 |

---

## Observations

### Logistic Regression
Performed well as a baseline model with good accuracy and moderate MCC score.

### Decision Tree
Captured non-linear relationships but slightly lower overall performance.

### KNN
Provided balanced performance but slower prediction for large datasets.

### Naive Bayes
Fast model but assumes feature independence, resulting in lower performance.

### Random Forest
Improved performance significantly due to ensemble learning.

### XGBoost
Best performing model with highest accuracy, AUC, and MCC score due to boosting technique.

---

## Project Structure

