# app.py
"""
WHAT THIS DOES:
Streamlit frontend for the Income Prediction API. Collects
inputs via a form, calls the LIVE deployed API (not a local
model), and renders the prediction + SHAP explanation visually.
"""
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "https://income-prediction-api-513976967636.us-central1.run.app"  # my real Cloud Run URL from before

st.set_page_config(page_title="Income Prediction", page_icon="💰", layout="centered")

st.title("Income Prediction with Explainability")
st.markdown(
    "Predicts whether someone earns **>$50K/year** from census-style attributes, "
    "using a Random Forest trained on the Adult Census Income dataset — "
    "with a live SHAP explanation for every prediction, not just a bare label."
)

# ------------------------------------------------------------------
# THE FORM — mirrors the PersonInput schema in api.py exactly
# ------------------------------------------------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 17, 90, 35)
        education_num = st.slider("Education level (1=lowest, 16=Doctorate)", 1, 16, 10)
        hours_per_week = st.slider("Hours worked per week", 1, 99, 40)
        capital_gain = st.number_input("Capital gain ($)", min_value=0, value=0, step=500)
        capital_loss = st.number_input("Capital loss ($)", min_value=0, value=0, step=500)

    with col2:
        workclass = st.selectbox("Work class", [
            "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
            "Local-gov", "State-gov", "Unknown",
        ])
        occupation = st.selectbox("Occupation", [
            "Exec-managerial", "Prof-specialty", "Tech-support", "Sales",
            "Craft-repair", "Adm-clerical", "Machine-op-inspct",
            "Farming-fishing", "Handlers-cleaners", "Other-service", "Unknown",
        ])
        marital_status = st.selectbox("Marital status", [
            "Married-civ-spouse", "Never-married", "Divorced",
            "Separated", "Widowed",
        ])
        relationship = st.selectbox("Relationship", [
            "Husband", "Wife", "Not-in-family", "Own-child", "Unmarried",
        ])
        sex = st.selectbox("Sex", ["Male", "Female"])
        race = st.selectbox("Race", [
            "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other",
        ])

    submitted = st.form_submit_button("Predict")

# ------------------------------------------------------------------
# ON SUBMIT: call the LIVE API, exactly like Tuesday's curl command
# ------------------------------------------------------------------
if submitted:
    payload = {
        "age": age,
        "workclass": workclass,
        "occupation": occupation,
        "education_num": education_num,
        "marital_status": marital_status,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "native_country": "United-States",
    }

    with st.spinner("Calling the live model..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout= (10, 60))
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API call failed: {e}")
            st.stop()

    # --- Prediction result ---
    prediction = result["prediction"]
    probability = result["probability_above_50k"]

    if prediction == ">50K":
        st.success(f"### Predicted: **{prediction}**  (probability: {probability:.1%})")
    else:
        st.info(f"### Predicted: **{prediction}**  (probability of >50K: {probability:.1%})")

    # --- SHAP explanation, rendered as a horizontal bar chart ---
    st.subheader("Why the model predicted this")
    factors = result["top_contributing_factors"]
    factor_df = pd.DataFrame(factors)
    factor_df = factor_df.sort_values("value")

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in factor_df["value"]]
    ax.barh(factor_df["feature"], factor_df["value"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution to prediction (SHAP value)")
    ax.set_title("Top 5 factors — green pushes toward >50K, red pushes toward <=50K")
    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "These are real SHAP values from the deployed model, computed fresh "
        "for this exact input — not a static explanation."
    )

st.markdown("---")
st.caption(
    "Model: Random Forest · Explainability: SHAP TreeExplainer · "
    "Served via FastAPI on Google Cloud Run · Data: UCI Adult Census Income"
)