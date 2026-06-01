# 📉 AI-Powered Customer Churn Prediction

## Overview

The AI-Powered Customer Churn Prediction Platform is a Machine Learning and Deep Learning project designed to predict whether a telecom customer is likely to leave a service provider. The application uses an Artificial Neural Network (ANN) trained on the Telco Customer Churn dataset and provides real-time churn risk predictions through an interactive Streamlit dashboard.

The platform helps telecom companies identify high-risk customers and take proactive retention measures to reduce customer loss and improve revenue.

---

## Features

### Customer Churn Prediction

* Predicts whether a customer is likely to churn.
* Generates churn probability and retention probability.

### Interactive Dashboard

* User-friendly Streamlit interface.
* Organized customer information and service details.
* Customer profile presets:

  * High Risk Customer
  * Medium Risk Customer
  * Low Risk Customer
  * Custom Input

### Risk Analysis

* Identifies customer risk factors.
* Explains why a customer may be likely to churn.

### Business Recommendations

* Provides actionable retention strategies.
* Helps businesses improve customer retention.

### Performance Metrics

* Model Accuracy: 77.4%
* ANN Architecture: 2 Hidden Layers
* Dataset Size: 7,043 Customers

---

## Dataset

Dataset: Telco Customer Churn Dataset

The dataset contains customer demographics, account information, subscribed services, billing details, and churn status.

Features include:

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract Type
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges

Target Variable:

* Churn

  * Yes
  * No

---

## Exploratory Data Analysis (EDA)

Key Findings:

### High-Risk Customers

* Month-to-Month Contract
* Low Tenure
* High Monthly Charges
* No Online Security
* No Technical Support

### Contract Analysis

| Contract Type  | Churn Rate |
| -------------- | ---------- |
| Month-to-Month | 42.71%     |
| One Year       | 11.27%     |
| Two Year       | 2.83%      |

### Business Insights

* Customers with shorter contracts are more likely to churn.
* Technical support and online security significantly reduce churn.
* New customers are more likely to leave compared to long-term customers.

---

## Machine Learning Pipeline

### Data Preprocessing

* Missing value handling
* Data cleaning
* One-Hot Encoding
* Feature Scaling using StandardScaler

### Model Development

Three ANN architectures were tested:

| Model           | Accuracy | Precision | Recall |
| --------------- | -------- | --------- | ------ |
| 2 Hidden Layers | 77.43%   | 59.59%    | 46.52% |
| 3 Hidden Layers | 76.72%   | 56.57%    | 52.94% |
| 5 Hidden Layers | 75.02%   | 53.11%    | 50.27% |

### Best Model

The ANN with 2 Hidden Layers achieved the best overall performance and was selected for deployment.

---

## Technology Stack

### Frontend

* Streamlit

### Backend

* Python

### Machine Learning

* TensorFlow / Keras
* Scikit-Learn

### Data Analysis

* Pandas
* NumPy

### Model Serialization

* Joblib

---

## Project Structure

```text
AI-Customer-Churn-Prediction/
│
├── app.py
├── churn_ann_model.keras
├── scaler.pkl
├── requirements.txt
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── README.md
└── AI_Customer_Churn_Prediction_ANN.ipynb
```

---


## Business Value

This platform enables telecom companies to:

* Predict customer churn before it happens.
* Reduce customer attrition.
* Improve customer satisfaction.
* Increase customer lifetime value.
* Optimize retention campaigns.

---

## Future Improvements

* XGBoost Model Comparison
* SHAP Explainable AI Integration
* Customer Segmentation
* Bulk Customer Prediction via CSV Upload
* Cloud Deployment with AWS or Azure
* Real-Time Monitoring Dashboard

