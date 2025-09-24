import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time

st.set_page_config(page_title="Health Insurance Recommender", layout="centered")

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("insurance_recommender_artifacts/model_pipeline.joblib")
    with open("insurance_recommender_artifacts/plan_catalog.json","r") as f:
        catalog = json.load(f)
    labels = pd.read_csv("insurance_recommender_artifacts/label_classes.csv", header=None)[0].tolist()
    return model, catalog, labels

model, catalog, labels = load_artifacts()

st.title("🩺 Smart Health Insurance Recommender")

# Track step
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.answers = {}

questions = [
    ("age", st.number_input, {"label":"What is your Age?", "min_value":18, "max_value":85, "value":30}),
    ("gender", st.radio, {"label":"Gender?", "options":["Male","Female"]}),
    ("smoker", st.radio, {"label":"Do you smoke?", "options":["No","Yes"]}),
    ("diabetes", st.radio, {"label":"Do you have Diabetes?", "options":["No","Yes"]}),
    ("hypertension", st.radio, {"label":"Do you have Hypertension?", "options":["No","Yes"]}),
    ("cardiac", st.radio, {"label":"Any Cardiac Condition?", "options":["No","Yes"]}),
    ("sum_insured_requested_lakh", st.slider, {"label":"Desired Sum Insured (₹ lakh)", "min_value":3, "max_value":100, "value":10})
]

if st.session_state.step < len(questions):
    key, widget, kwargs = questions[st.session_state.step]
    st.session_state.answers[key] = widget(**kwargs)

    if st.button("Next"):
        st.session_state.step += 1
        st.experimental_rerun()
else:
    st.write("🔍 Analyzing your profile...")
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i+1)

    # Convert Yes/No to 0/1
    X = pd.DataFrame([{
        "age": st.session_state.answers.get("age",30),
        "gender": st.session_state.answers.get("gender","Male"),
        "smoker": 1 if st.session_state.answers.get("smoker")=="Yes" else 0,
        "diabetes": 1 if st.session_state.answers.get("diabetes")=="Yes" else 0,
        "hypertension": 1 if st.session_state.answers.get("hypertension")=="Yes" else 0,
        "cardiac": 1 if st.session_state.answers.get("cardiac")=="Yes" else 0,
        "sum_insured_requested_lakh": st.session_state.answers.get("sum_insured_requested_lakh",10),
        "risk_score": 0  # placeholder
    }])

    proba = model.predict_proba(X)[0]
    pred_idx = np.argsort(proba)[::-1]
    top3 = [labels[i] for i in pred_idx[:3]]

    st.success("✅ We found the best plans for you!")

    for rank, cat in enumerate(top3, start=1):
        rows = [r for r in catalog if r["plan_category"]==cat]
        examples = rows[0]["example_products"] if rows else "—"
        if rank==1:
            st.markdown(f"### 🥇 Highly Recommended: **{cat}**")
        elif rank==2:
            st.markdown(f"### 🥈 Good Fit: **{cat}**")
        else:
            st.markdown(f"### 🥉 Consider: **{cat}**")
        st.caption(f"Examples: {examples}")
