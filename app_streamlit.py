import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Smart Health Insurance Recommender", layout="centered")

# ----------------- Load Artifacts -----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("insurance_recommender_artifacts/model_pipeline.joblib")
    with open("insurance_recommender_artifacts/plan_catalog.json", "r") as f:
        catalog = json.load(f)
    labels = pd.read_csv("insurance_recommender_artifacts/label_classes.csv", header=None)[0].tolist()
    return model, catalog, labels

model, catalog, labels = load_artifacts()

# ----------------- Wizard State -----------------
if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}

# ----------------- Questions -----------------
questions = [
    ("age", st.number_input, {"label":"📅 Age", "min_value":18, "max_value":85, "value":30}),
    ("gender", st.radio, {"label":"⚧ Gender", "options":["Male","Female"]}),
    ("marital_status", st.radio, {"label":"💍 Marital Status", "options":["Single","Married","Divorced","Widowed"]}),
    ("city_tier", st.selectbox, {"label":"🏙️ City Tier", "options":[1,2,3]}),
    ("state", st.selectbox, {"label":"🗺️ State (short code)", "options":["DL","MH","KA","TN","GJ","RJ","UP","WB","PB","TS","KL","HR","MP","AP","UK"]}),
    ("pincode", st.number_input, {"label":"📮 Pincode", "min_value":110001, "max_value":695615, "value":560001}),

    ("occupation", st.selectbox, {"label":"👔 Occupation", "options":["Salaried","Self-Employed","Student","Retired","Homemaker"]}),
    ("education", st.selectbox, {"label":"🎓 Education", "options":["High School","Graduate","Postgraduate","Diploma"]}),
    ("annual_income_lakh", st.slider, {"label":"💰 Annual Income (₹ lakh)", "min_value":1, "max_value":80, "value":12}),
    ("smoker", st.radio, {"label":"🚬 Smoker", "options":["No","Yes"]}),
    ("alcohol", st.radio, {"label":"🍷 Alcohol Use", "options":["No","Yes"]}),
    ("bmi", st.slider, {"label":"⚖️ BMI", "min_value":16.0, "max_value":45.0, "value":24.5, "step":0.1}),
    ("steps_per_day", st.slider, {"label":"🚶 Steps per day", "min_value":500, "max_value":20000, "value":6000, "step":500}),
    ("diabetes", st.radio, {"label":"🩸 Diabetes", "options":["No","Yes"]}),
    ("hypertension", st.radio, {"label":"💓 Hypertension", "options":["No","Yes"]}),
    ("cardiac", st.radio, {"label":"❤️ Cardiac condition", "options":["No","Yes"]}),
    ("asthma", st.radio, {"label":"🌬️ Asthma", "options":["No","Yes"]}),
    ("cancer_history", st.radio, {"label":"🎗️ Cancer history", "options":["No","Yes"]}),
    ("prior_hospitalizations_3y", st.slider, {"label":"🏥 Hospitalizations (3y)", "min_value":0, "max_value":10, "value":0}),
    ("claims_history", st.slider, {"label":"📑 Claims count (3y)", "min_value":0, "max_value":10, "value":0}),
    ("claim_amount_last3y_lakh", st.slider, {"label":"💵 Claim amount (3y, ₹ lakh)", "min_value":0, "max_value":100, "value":0}),
    ("num_dependents", st.slider, {"label":"👨‍👩‍👧‍👦 Dependents", "min_value":0, "max_value":10, "value":0}),
    ("is_pregnant", st.radio, {"label":"🤰 Currently Pregnant (if Female)", "options":["No","Yes"]}),
    ("planning_maternity_2y", st.radio, {"label":"👶 Planning Maternity in 2y?", "options":["No","Yes"]}),
    ("sum_insured_requested_lakh", st.slider, {"label":"🛡️ Desired Sum Insured (₹ lakh)", "min_value":3, "max_value":100, "value":10}),
    ("existing_employer_cover_lakh", st.slider, {"label":"🏢 Employer Cover (₹ lakh)", "min_value":0, "max_value":20, "value":0}),
    ("opd_need", st.radio, {"label":"💊 Need OPD?", "options":["No","Yes"]}),
    ("dental_need", st.radio, {"label":"🦷 Need Dental?", "options":["No","Yes"]}),
    ("high_travel", st.radio, {"label":"✈️ High Travel?", "options":["No","Yes"]}),
    ("risk_aversion", st.radio, {"label":"⚠️ Risk Appetite", "options":["Low","Medium","High"]})
]

# ----------------- UI Rendering -----------------
st.title("🩺 SureMate AI Driven Insurance Platform")

total_steps = len(questions)
progress = st.session_state.step / total_steps
st.progress(progress)
st.caption(f"📊 Step {st.session_state.step+1} of {total_steps}" if st.session_state.step < total_steps else "📊 Review & Results")

# ----------------- Ask Questions -----------------
if st.session_state.step < len(questions):
    key, widget, kwargs = questions[st.session_state.step]

    # remove conflicting keys (important!)
    safe_kwargs = {k: v for k, v in kwargs.items() if k not in ["value", "index", "options"]}

    if widget in [st.radio, st.selectbox]:
        options = kwargs.get("options", [])
        default_val = st.session_state.answers.get(key, kwargs.get("value", options[0] if options else None))
        if default_val in options:
            idx = options.index(default_val)
        else:
            idx = 0
        st.session_state.answers[key] = widget(**safe_kwargs, options=options, index=idx)
    else:
        st.session_state.answers[key] = widget(
            **safe_kwargs,
            value=st.session_state.answers.get(key, kwargs.get("value"))
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Back") and st.session_state.step > 0:
            st.session_state.step -= 1
            st.rerun()
    with col2:
        if st.button("Next ➡️"):
            st.session_state.step += 1
            st.rerun()

# ----------------- Review & Prediction -----------------
else:
    st.subheader("🔎 Review Your Details")
    st.json(st.session_state.answers)

    if st.button("✏️ Edit Answers"):
        st.session_state.step = 0
        st.rerun()

    if st.button("✅ Get Recommendations"):
        X = pd.DataFrame([{
            "age": st.session_state.answers["age"],
            "gender": st.session_state.answers["gender"],
            "marital_status": st.session_state.answers["marital_status"],
            "city_tier": st.session_state.answers["city_tier"],
            "state": st.session_state.answers["state"],
            "pincode": st.session_state.answers["pincode"],
            "occupation": st.session_state.answers["occupation"],
            "education": st.session_state.answers["education"],
            "annual_income_lakh": st.session_state.answers["annual_income_lakh"],
            "smoker": 1 if st.session_state.answers["smoker"]=="Yes" else 0,
            "alcohol": 1 if st.session_state.answers["alcohol"]=="Yes" else 0,
            "bmi": st.session_state.answers["bmi"],
            "steps_per_day": st.session_state.answers["steps_per_day"],
            "diabetes": 1 if st.session_state.answers["diabetes"]=="Yes" else 0,
            "hypertension": 1 if st.session_state.answers["hypertension"]=="Yes" else 0,
            "cardiac": 1 if st.session_state.answers["cardiac"]=="Yes" else 0,
            "asthma": 1 if st.session_state.answers["asthma"]=="Yes" else 0,
            "cancer_history": 1 if st.session_state.answers["cancer_history"]=="Yes" else 0,
            "prior_hospitalizations_3y": st.session_state.answers["prior_hospitalizations_3y"],
            "claims_history": st.session_state.answers["claims_history"],
            "claim_amount_last3y_lakh": st.session_state.answers["claim_amount_last3y_lakh"],
            "num_dependents": st.session_state.answers["num_dependents"],
            "is_pregnant": 1 if st.session_state.answers["is_pregnant"]=="Yes" else 0,
            "planning_maternity_2y": 1 if st.session_state.answers["planning_maternity_2y"]=="Yes" else 0,
            "sum_insured_requested_lakh": st.session_state.answers["sum_insured_requested_lakh"],
            "existing_employer_cover_lakh": st.session_state.answers["existing_employer_cover_lakh"],
            "opd_need": 1 if st.session_state.answers["opd_need"]=="Yes" else 0,
            "dental_need": 1 if st.session_state.answers["dental_need"]=="Yes" else 0,
            "high_travel": 1 if st.session_state.answers["high_travel"]=="Yes" else 0,
            "risk_aversion": st.session_state.answers["risk_aversion"],
        }])

        # risk score heuristic
        risk_score = (
            0.02*X["age"] + 0.8*X["diabetes"] + 0.8*X["hypertension"] + 1.2*X["cardiac"] +
            1.5*X["cancer_history"] + 0.5*X["asthma"] + 0.6*X["smoker"] + 0.3*X["alcohol"] +
            0.03*X["prior_hospitalizations_3y"] + 0.04*X["claims_history"] + (X["bmi"]>=30).astype(int)
        )
        risk_score = ((risk_score - risk_score.min())/(risk_score.max()-risk_score.min()+1e-6))*100
        X["risk_score"] = risk_score

        # prediction
        proba = model.predict_proba(X)[0]
        pred_idx = np.argsort(proba)[::-1]
        top3 = [(labels[i], float(proba[i])) for i in pred_idx[:3]]

        def examples_for(cat):
            rows = [r for r in catalog if r["plan_category"]==cat]
            return rows[0]["example_products"] if rows else "—"

        st.subheader("🏆 Your Recommended Plans")
        medals = ["🥇 Highly Recommended", "🥈 Good Fit", "🥉 Consider"]
        for i, (cat, _) in enumerate(top3):
            st.markdown(f"**{medals[i]}: {cat}**")
            st.caption(f"Examples: {examples_for(cat)}")
            st.write("---")

    if st.button("🔄 Start Over"):
        st.session_state.step = 0
        st.session_state.answers = {}
        st.rerun()



