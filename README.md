# 🎓 ISPDSS — Intelligent Student Performance Decision Support System

An AI-powered Intelligent Decision Support System (IDSS) that predicts student academic
performance and recommends actionable interventions — built to help teachers and academic
advisors identify at-risk students early, instead of after final grades are already in.

The system combines **machine learning** (5 classification + 5 regression algorithms,
benchmarked and cross-validated) with a **rule-based knowledge layer** that turns raw
predictions into explainable, human-readable recommendations.

---

## ✨ Features

- **Dual prediction**: Grade Category (A–F) via classification, and numeric GPA (0.0–4.0) via regression
- **5 algorithms per task, benchmarked head-to-head**: Logistic Regression, Random Forest, SVM, KNN, XGBoost (classification) · Linear Regression, Random Forest, SVR, XGBoost, Gradient Boosting (regression)
- **Hybrid knowledge representation**: ML predictions + a symbolic IF-THEN rule engine for explainable risk flags and recommended interventions
- **Full evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, RMSE, MAE, R², plus 5-fold cross-validation for robustness
- **Interactive dashboard**: Streamlit app with Predict & Advise, Model Comparison, and About tabs
- **End-to-end reproducibility**: one notebook covers EDA → training → evaluation; one file deploys the app

---

## 📸 Project Dashboard Screenshots

![Dashboard]()
![Prediction&Advisetab]()
![Predictedinput]()
![Predictedoutput]()
![ModelComparison]()
![info]()
<!--
  Add screenshots of the running Streamlit dashboard here.
  Suggested shots: the Predict & Advise tab (input form + results),
  the Model Comparison tab, and the About tab.
  Example:
  ![Predict & Advise](screenshots/predict_and_advise.png)
  ![Model Comparison](screenshots/model_comparison.png)
-->

---

## 🗂️ Project Structure

```
ISPDSS-Intelligent_Student_Performance_Decision_SupportSystem/
├── notebook.ipynb                   # Full pipeline: EDA, preprocessing, training, evaluation, CV
├── app.py                           # Deployable Streamlit dashboard
├── data/
│   └── Student_performance_data__.csv
├── models/                          # Trained models + scalers (.joblib)
│   ├── clf_Logistic_Regression.joblib
│   ├── clf_Random_Forest.joblib
│   ├── clf_SVM_RBF.joblib
│   ├── clf_K-Nearest_Neighbors.joblib
│   ├── clf_XGBoost.joblib
│   ├── reg_Linear_Regression.joblib
│   ├── reg_Random_Forest_Regressor.joblib
│   ├── reg_SVR_RBF.joblib
│   ├── reg_XGBoost_Regressor.joblib
│   ├── reg_Gradient_Boosting.joblib
│   ├── clf_scaler.joblib / reg_scaler.joblib
│   └── feature_cols.joblib
└── outputs/                         # Result tables (CSV) and charts (PNG)
|   ├── classification_results.csv
|   ├── classification_cv_results.csv
|   ├── regression_results.csv
|   ├── regression_cv_results.csv
|   ├── confusion_matrix_rf.png
|   ├── eda_correlation.png
|   └── grade_gpa_distribution.png
└── demo                            # Project dashboard screenshots
    ├── classification_results.csv
    ├── classification_cv_results.csv
    ├── regression_results.csv
    ├── regression_cv_results.csv
    ├── confusion_matrix_rf.png
    ├── eda_correlation.png

```

---

## 📊 Dataset

[Students Performance Dataset](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset)
(Kaggle, Rabie El Kharoua) — 2,392 students, 15 attributes: demographics, study habits,
parental involvement, extracurricular activity, attendance, and two targets: `GPA`
(regression) and `GradeClass` (classification, A–F).

> Note: `GradeClass` is imbalanced — "F" is ~50.6% of records, "A" only ~4.5%. This is
> explicitly accounted for in evaluation (macro-averaged metrics, stratified splits, CV).

---

## 🧠 Knowledge Representation

1. **Structured data layer** — tabular features (demographics, study time, absences,
   tutoring, parental support, extracurricular engagement) feed the ML models directly.
2. **Rule-based (symbolic) layer** — domain IF-THEN rules generate explainable risk flags
   and recommendations independent of the ML models, e.g.:
   - `Absences > 20 AND StudyTimeWeekly < 5` → High risk, recommend structured study plan
   - `Tutoring = No AND ParentalSupport ≤ 1` → Low support network, recommend tutoring
   - `Extracurricular = No AND Absences > 15` → Disengagement signal, recommend counseling

---

## 📈 Results Summary

**Classification (Grade Category A–F)** — test set (20% hold-out):

| Model               | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (macro) |
| ------------------- | -------- | ----------------- | -------------- | ---------- | --------------- |
| Logistic Regression | 0.727    | 0.573             | 0.533          | 0.540      | 0.869           |
| SVM (RBF)           | 0.716    | 0.651             | 0.543          | 0.554      | 0.868           |
| Random Forest       | 0.716    | 0.596             | 0.543          | 0.548      | 0.859           |
| XGBoost             | 0.706    | 0.571             | 0.544          | 0.550      | 0.851           |
| K-Nearest Neighbors | 0.608    | 0.459             | 0.372          | 0.372      | 0.788           |

Best generalization under 5-fold CV (macro F1): **Random Forest (0.564)** and **XGBoost (0.561)**.

**Regression (GPA, 0–4 scale)** — test set:

| Model                   | RMSE  | MAE   | R²    |
| ----------------------- | ----- | ----- | ----- |
| Linear Regression       | 0.197 | 0.155 | 0.953 |
| Gradient Boosting       | 0.209 | 0.164 | 0.947 |
| XGBoost Regressor       | 0.218 | 0.174 | 0.943 |
| SVR (RBF)               | 0.232 | 0.183 | 0.935 |
| Random Forest Regressor | 0.238 | 0.186 | 0.931 |

Full metrics, confusion matrices, and charts are in `outputs/`.

---

## 🚀 Getting Started

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd ISPDSS-Intelligent_Student_Performance_Decision_SupportSystem
pip install pandas numpy scikit-learn xgboost matplotlib joblib jupyter streamlit
```

### 2. (Optional) Re-run the full analysis

Models are already trained and saved in `models/`, but to reproduce or modify the pipeline:

```bash
jupyter notebook notebook.ipynb
```

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

---

## 🛠️ Tech Stack

Python · scikit-learn · XGBoost · pandas / NumPy · Streamlit · Jupyter Notebook · matplotlib

---

## ⚠️ Disclaimer

This is a simulated decision-support prototype built for educational purposes. It does not
replace professional academic or psychological judgment.

---

## 📄 License

Add your preferred license here (e.g., MIT).
