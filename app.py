import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUT_DIR = os.path.join(BASE_DIR, 'outputs')

GRADE_LABELS = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'F'
}

st.set_page_config(page_title="Student Performance DSS", layout="wide")

# Recommendation engines
# Rule based risk assessment
def assess_risk_rules(student):
    flags = []
    if student["Absences"] > 20 and student["StudyTimeWeekly"] < 5:
        flags.append((
            "High Risk: chronic absence + low study time",
            "Immediate advisor meeting | structured study plan + attendance monitoring."
        ))
    if student["Tutoring"] == 0 and student["ParentalSupport"] <= 1:
        flags.append((
            "Low support network",
            "Recommend tutoring enrollment | notify guardians about support resources."
        ))
    if student["Extracurricular"] == 0 and student["Absences"] > 15:
        flags.append((
            "Disengagement signal",
            "Recommend counseling check-in and engagement activities."
        ))
    if student["StudyTimeWeekly"] < 3:
        flags.append((
            "Very low study time",
            "Introduce a weekly study-hours goal and progress check-ins."
        ))
    if not flags:
        flags.append(("No major risk flags", "Continue current support | monitor periodically."))
    return flags

# Priority recommendation
def generate_recommendation(student, predicted_gpa, predicted_grade_label):
    flags = assess_risk_rules(student)
    if predicted_grade_label in ("F", "D"):
        overall = "HIGH PRIORITY"
    elif predicted_grade_label == "C":
        overall = "MODERATE"
    else:
        overall = "LOW PRIORITY"
    return {
        "predicted_gpa": round(float(predicted_gpa), 2),
        "predicted_grade": predicted_grade_label,
        "overall_status": overall,
        "risk_flags": flags,
    }

# Load model
def load_artifacts():
    clf = joblib.load(f"{MODELS_DIR}/clf_Random_Forest.joblib")
    reg = joblib.load(f"{MODELS_DIR}/reg_Linear_Regression.joblib")
    clf_scaler = joblib.load(f"{MODELS_DIR}/clf_scaler.joblib")
    reg_scaler = joblib.load(f"{MODELS_DIR}/reg_scaler.joblib")
    feature_cols = joblib.load(f"{MODELS_DIR}/feature_cols.joblib")

    return clf, reg, clf_scaler, reg_scaler, feature_cols
clf, reg, clf_scaler, reg_scaler, feature_cols = load_artifacts()

# Readable label maps
ETHNICITY_LABELS = {0: "Caucasian", 1: "African American", 2: "Asian", 3: "Other"}
PARENTAL_EDU_LABELS = {0: "None", 1: "High School", 2: "Some College", 3: "Bachelor's", 4: "Higher"}
PARENTAL_SUPPORT_LABELS = {0: "None", 1: "Low", 2: "Moderate", 3: "High", 4: "Very High"}

# Page elements
st.title("Intelligent Student Performance Decision Support System")
st.caption(
    "Predicts GPA and grade risk category from early-term indicators, "
    "and recommends interventions using a hybrid ML + rule-based engine.")

tab1, tab2, tab3 = st.tabs(["Predict & Advise", "Model Comparison", "Info"])

# tab 1: Prediction and recommendation
with tab1:
    st.subheader("Student Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 15, 18, 16)
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
        ethnicity = st.selectbox("Ethnicity", options=[0, 1, 2, 3], format_func=lambda x: ETHNICITY_LABELS[x])
        parental_edu = st.selectbox("Parental Education", options=[0, 1, 2, 3, 4],
                                     format_func=lambda x: PARENTAL_EDU_LABELS[x])

    with col2:
        study_time = st.slider("Weekly Study Time (hours)", 0.0, 20.0, 8.0, 0.5)
        absences = st.slider("Absences (this term)", 0, 30, 10)
        tutoring = st.selectbox("Tutoring", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        parental_support = st.selectbox("Parental Support", options=[0, 1, 2, 3, 4],
                                         format_func=lambda x: PARENTAL_SUPPORT_LABELS[x])

    with col3:
        extracurricular = st.selectbox("Extracurricular", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        sports = st.selectbox("Sports", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        music = st.selectbox("Music", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        volunteering = st.selectbox("Volunteering", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    if st.button("Generate Prediction & Recommendation", type="primary"):
        attendance_rate = 1 - (absences / 180)
        engagement_score = extracurricular + sports + music + volunteering

        raw_features = pd.DataFrame([{
            "Age": age, "Gender": gender, "Ethnicity": ethnicity,
            "ParentalEducation": parental_edu, "StudyTimeWeekly": study_time,
            "Absences": absences, "Tutoring": tutoring, "ParentalSupport": parental_support,
            "Extracurricular": extracurricular, "Sports": sports, "Music": music,
            "Volunteering": volunteering, "AttendanceRate": attendance_rate,
            "EngagementScore": engagement_score
        }])[feature_cols]

        X_clf = clf_scaler.transform(raw_features)
        X_reg = reg_scaler.transform(raw_features)

        pred_class = clf.predict(X_clf)[0]
        pred_grade = GRADE_LABELS[pred_class]
        pred_gpa = reg.predict(X_reg)[0]
        pred_gpa = max(0.0, min(4.0, pred_gpa))

        student_dict = {
            "StudyTimeWeekly": study_time, "Absences": absences, "Tutoring": tutoring,
            "ParentalSupport": parental_support, "Extracurricular": extracurricular
        }
        result = generate_recommendation(student_dict, pred_gpa, pred_grade)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted GPA", f"{result['predicted_gpa']} / 4.0")
        c2.metric("Predicted Grade Category", result["predicted_grade"])
        c3.metric("Status", result["overall_status"])

        st.subheader("Recommended Actions")
        for flag, rec in result["risk_flags"]:
            st.warning(f"**{flag}**\n\n→ {rec}")

# tab 2: Model comparison
with tab2:
    st.subheader("Classification Models — Predicting Grade Category (A-F)")
    try:
        clf_results = pd.read_csv(f"{OUT_DIR}/classification_results.csv")
        st.dataframe(clf_results, use_container_width=True)
        clf_cv = pd.read_csv(f"{OUT_DIR}/classification_cv_results.csv")
        st.caption("5-fold cross-validation (macro F1)")
        st.dataframe(clf_cv, use_container_width=True)
    except FileNotFoundError:
        st.info("Notebook (Student_Performance_DSS.ipynb) must be run first to generate these results.")

    st.subheader("Regression Models — Predicting GPA")
    try:
        reg_results = pd.read_csv(f"{OUT_DIR}/regression_results.csv")
        st.dataframe(reg_results, use_container_width=True)
        reg_cv = pd.read_csv(f"{OUT_DIR}/regression_cv_results.csv")
        st.caption("5-fold cross-validation (R²)")
        st.dataframe(reg_cv, use_container_width=True)
    except FileNotFoundError:
        st.info("Notebook (Student_Performance_DSS.ipynb) must be run first to generate these results.")

# tab 3: Info
with tab3:
    st.markdown("""
    ### About this system
    This is an **Intelligent Student Performance Decision Support System (ISPDSS)** for early
    identification of at-risk students.

    **Architecture:**
    - **Data layer**: 2,392 student records (demographics, study habits, support, engagement)
    - **ML layer**: 5 classification algorithms (predicting grade category A-F) and
      5 regression algorithms (predicting GPA), trained and compared in
      `notebook.ipynb`.
    - **Rule layer**: Symbolic IF-THEN rules that translate raw student
      indicators into explainable risk flags and recommended interventions.
    - **Decision layer/UI**: combines ML prediction + rules into a single
      actionable advisory report for teachers/advisors.
    """)