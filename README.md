# Diabetes 30-Day Readmission Risk App

A **Streamlit web application** that predicts whether a **diabetic patient** is likely to be **readmitted within 30 days** after hospital discharge.  
This project combines healthcare knowledge and data science to support hospitals in improving patient outcomes and resource allocation.

---

##  Overview

The app analyzes patient details — including demographics, hospital stay length, medication changes, and diabetes treatment — to estimate their risk of being readmitted shortly after discharge.

The prediction is powered by a **machine learning model (Logistic Regression)** trained on over **100,000 hospital encounters**.

---

##  Project Workflow

1. **Data Understanding & Cleaning**
   - Removed irrelevant or high-missing-value columns.
   - Encoded categorical features.
   - Handled missing values and created a binary target variable (`readmit_30`).

2. **Exploratory Data Analysis (EDA)**
   - Visualized relationships between age, race, insulin use, hospital stay length, and readmission.
   - Found that longer hospital stays and insulin adjustments often increase readmission risk.

3. **Model Building**
   - Model used: Logistic Regression
   - Target variable: Readmission within 30 days (`readmit_30`)
   - ROC-AUC Score: **0.65**
   - Balanced performance without oversampling (reflecting real-world hospital patterns)

4. **Web App Development**
   - Built using Streamlit.
   - Takes user input (patient details) and outputs:
     - Probability of 30-day readmission
     - Risk level: 🟢 Low | 🟡 Medium | 🔴 High

---

## Key Features

- **Interactive Prediction Interface**: Enter patient details and instantly see readmission risk.
- **Probability-Based Output**: Displays confidence score and color-coded risk.
- **Realistic Model Design**: Keeps natural data imbalance to reflect hospital data distribution.
- **Clean UI**: Built with Streamlit, optimized for clinicians and researchers.

---

## Features Used in Modeling

| Category | Features |
|-----------|-----------|
|  Demographics | Race, Gender, Age |
|  Hospital Info | Admission Type, Discharge Type, Admission Source |
|  Stay Details | Time in Hospital |
|  Clinical Data | Number of Lab Procedures, Procedures, Medications, Diagnoses |
|  Visit History | Outpatient, Emergency, and Inpatient Visits |
|  Diabetes Control | Insulin, Change, DiabetesMed |
|  Diagnosis Info | Diag_1, Diag_2, Diag_3 |

---

##  Model Insights

- Most diabetic patients are **not readmitted** within 30 days (~89%).
- Patients with **longer hospital stays** or **medication changes** tend to have higher risk.
- The app’s results are **probability-based**, making it more flexible and interpretable.

---

##  Why the Data Imbalance Was Kept

- The imbalance reflects **real hospital situations** — only about 1 in 10 patients return.
- Balancing artificially might cause **false alarms**.
- The app handles imbalance by using **probabilities and ROC-AUC scoring**, not raw counts.

---

##  Deployment Guide

###  Local Setup

Clone the repo:

```bash
git clone https://github.com/<your-username>/diabetes-readmission-project.git
cd diabetes-readmission-project

