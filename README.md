# 📉 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Keras](https://img.shields.io/badge/Keras--Tuner-HPO-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-2E7D32?style=for-the-badge)

End-to-end machine learning and deep learning project to predict telecom customer churn using billing, tenure, and service subscription data. Covers the full pipeline from data cleaning through model deployment with saved artifacts.

---

## 📌 Problem Statement

Customer churn — when a subscriber stops using a service — is one of the most costly problems in the telecom industry. Acquiring a new customer costs 5–7× more than retaining an existing one. This project builds a predictive model to identify customers at high risk of churning so retention teams can act proactively.

---

## 📂 Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | Telecom Customer Churn Dataset |
| **Records** | 7,043 customers |
| **Features** | 21 (demographics, services, billing, contract) |
| **Target** | `Churn` — Yes (1) / No (0) |

### Key Features

| Feature | Description |
|---------|-------------|
| `tenure` | Number of months the customer has been with the company |
| `MonthlyCharges` | Current monthly billing amount |
| `TotalCharges` | Total amount billed to date |
| `Contract` | Month-to-month / One year / Two year |
| `PaymentMethod` | Electronic check / Mailed check / Bank transfer / Credit card |
| `InternetService` | DSL / Fiber optic / No |
| `SeniorCitizen` | Whether the customer is a senior citizen |
| `Churn` | **Target** — whether the customer left the company |

---

## 🔁 Project Pipeline

```
Raw CSV
   │
   ▼
Data Cleaning          ← Fix TotalCharges blanks, encode Churn to 0/1
   │
   ▼
Data Manipulation      ← Filter subsets by business conditions
   │
   ▼
Visualization          ← Internet service, tenure distribution, scatter & boxplots
   │
   ▼
Linear Regression      ← Predict MonthlyCharges from tenure (baseline analysis)
   │
   ▼
Logistic Regression    ← Simple (1 feature) → Multiple (2 features)
   │
   ▼
Decision Tree          ← Single feature churn classification
   │
   ▼
Random Forest + RFE    ← Select top 5 features from 19
   │
   ▼
Deep Neural Network    ← Baseline DNN → Keras Tuner HPO → Best model
   │
   ▼
Save Artifacts         ← best_churn_model.h5 | scaler.pkl | selected_features.pkl
   │
   ▼
Prediction Pipeline    ← Load & predict on new customer input
```

---

## 🔬 Data Cleaning Notes

| Issue | Fix |
|-------|-----|
| `Churn` stored as `"Yes"/"No"` string | Mapped to binary `1 / 0` |
| 11 rows with blank `TotalCharges` (where `tenure = 0`) | Recalculated as `MonthlyCharges × tenure` |
| No duplicate rows | No action required |
| No remaining nulls post-fix | Dataset clean for modelling |

---

## 🤖 Models & Results

### 1. Linear Regression
- **Task:** Predict `MonthlyCharges` from `tenure`
- **Split:** 70 / 30
- **Metrics:** MAE, RMSE, R²
- **Purpose:** Establish billing trend baseline

### 2. Logistic Regression

| Variant | Features | Split | Notes |
|---------|----------|-------|-------|
| Simple | `MonthlyCharges` | 65/35 | Single predictor baseline |
| Multiple | `tenure` + `MonthlyCharges` | 80/20 | Two-feature model |

### 3. Decision Tree
- **Feature:** `tenure`
- **Split:** 80 / 20
- **Output:** Confusion matrix + classification report

### 4. Random Forest + RFE Feature Selection
- **Estimators:** 150
- **RFE output:** Top 5 features selected from 19
- **Feature importance** plotted via horizontal bar chart

### 5. Deep Neural Network — Baseline

```
Input (5) → Dense(64, relu) → Dense(32, relu) → Dense(16, relu) → Dense(8, relu) → Dense(1, sigmoid)
Optimizer: Adam | Loss: Binary Crossentropy | Epochs: 50 | Batch: 32
```

### 6. Deep Neural Network — Keras Tuner (Best Model)

```
Search Strategy : RandomSearch
Tuned params    : units_1 ∈ [32, 64, 128] | second_layer ∈ [True, False] | units_2 ∈ [16, 32]
Objective       : val_accuracy
Max trials      : 5
Early stopping  : patience = 3 (val_loss)
Final epochs    : 50
```

---

## 📦 Saved Artifacts

| File | Description |
|------|-------------|
| `best_churn_model.h5` | Best tuned Keras model |
| `scaler.pkl` | Fitted StandardScaler for inference |
| `selected_features.pkl` | List of 5 RFE-selected feature names |

---

## 🗂️ Project Structure

```
customer-churn-prediction/
│
├── customer_churn_improved.ipynb   ← Main notebook
├── customer_churn.csv              ← Dataset
├── best_churn_model.h5             ← Saved model
├── scaler.pkl                      ← Saved scaler
├── selected_features.pkl           ← Saved feature list
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/givishalt/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook customer_churn_improved.ipynb
```

### Requirements

```
pandas
numpy
seaborn
matplotlib
plotly
scikit-learn
imbalanced-learn
tensorflow
keras-tuner
pickle-mixin
jupyter
```

---

## 🔮 Future Scope

- [ ] Wrap prediction pipeline into a **FastAPI REST endpoint**
- [ ] Build a **Streamlit app** for real-time churn probability lookup
- [ ] Add **SHAP explainability** to understand per-customer churn drivers
- [ ] Experiment with **XGBoost / LightGBM** for tabular performance comparison
- [ ] Incorporate **class imbalance handling** (SMOTE) into the DNN pipeline

---

## 👤 Author

**Vishal**
- 📧 vishal9681032@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/vishal-793170396/)
- 🐙 [GitHub](https://github.com/givishalt)

---

## 📄 License

This project is licensed under the **MIT License**.
