import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="centered"
)


# ============================================================
# LOAD MODEL, SCALER AND FEATURE NAMES
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("models/loan_model.pkl")

    scaler = joblib.load("models/scaler.pkl")

    with open("models/feature_names.json", "r") as file:
        feature_names = json.load(file)

    return model, scaler, feature_names


model, scaler, feature_names = load_artifacts()


# ============================================================
# TITLE
# ============================================================

st.title("💰 Loan Approval Predictor")

st.write(
    "Enter the applicant's information below to predict "
    "whether the loan is likely to be approved."
)

st.divider()


# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.subheader("👤 Applicant Information")

col1, col2 = st.columns(2)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )


with col2:

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000,
        step=100
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=150.0,
        step=10.0
    )

    loan_term = st.number_input(
        "Loan Amount Term",
        min_value=12,
        value=360,
        step=12
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0],
        format_func=lambda x: "Good (1)" if x == 1.0 else "Bad (0)"
    )


property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔮 Predict Loan Approval",
    use_container_width=True
):

    # --------------------------------------------------------
    # CREATE INPUT DATA
    # --------------------------------------------------------

    input_data = {

        "ApplicantIncome":
            applicant_income,

        "CoapplicantIncome":
            coapplicant_income,

        "LoanAmount":
            loan_amount,

        "Loan_Amount_Term":
            loan_term,

        "Credit_History":
            credit_history,

        "Gender_Male":
            1 if gender == "Male" else 0,

        "Married_Yes":
            1 if married == "Yes" else 0,

        "Dependents_1":
            1 if dependents == "1" else 0,

        "Dependents_2":
            1 if dependents == "2" else 0,

        "Dependents_3+":
            1 if dependents == "3+" else 0,

        "Education_Not Graduate":
            1 if education == "Not Graduate" else 0,

        "Self_Employed_Yes":
            1 if self_employed == "Yes" else 0,

        "Property_Area_Semiurban":
            1 if property_area == "Semiurban" else 0,

        "Property_Area_Urban":
            1 if property_area == "Urban" else 0
    }


    # --------------------------------------------------------
    # CONVERT TO DATAFRAME
    # --------------------------------------------------------

    input_df = pd.DataFrame([input_data])


    # --------------------------------------------------------
    # ENSURE EXACT FEATURE ORDER
    # --------------------------------------------------------

    input_df = input_df[feature_names]


    # --------------------------------------------------------
    # SCALE INPUT
    # --------------------------------------------------------

    input_scaled = scaler.transform(input_df)


    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_scaled)[0]


    # --------------------------------------------------------
    # PREDICTION PROBABILITY
    # --------------------------------------------------------

    probability = model.predict_proba(input_scaled)[0]

    approval_probability = probability[1] * 100

    rejection_probability = probability[0] * 100


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()

    st.subheader("📊 Prediction Result")


    if prediction == 1:

        st.success(
            "✅ LOAN APPROVED"
        )

        st.metric(
            "Approval Probability",
            f"{approval_probability:.2f}%"
        )

    else:

        st.error(
            "❌ LOAN NOT APPROVED"
        )

        st.metric(
            "Approval Probability",
            f"{approval_probability:.2f}%"
        )


    # --------------------------------------------------------
    # PROBABILITY DETAILS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Approval",
            f"{approval_probability:.2f}%"
        )

    with col2:

        st.metric(
            "Rejection",
            f"{rejection_probability:.2f}%"
        )


    # --------------------------------------------------------
    # SHOW PROCESSED INPUT
    # --------------------------------------------------------

    with st.expander("🔍 View Model Input"):

        st.dataframe(
            input_df,
            use_container_width=True
        )


    # --------------------------------------------------------
    # SHOW MODEL INFORMATION
    # --------------------------------------------------------

    with st.expander("🤖 Model Information"):

        st.write(
            "Model: Logistic Regression"
        )

        st.write(
            f"Number of features: {len(feature_names)}"
        )

        st.write(
            "Prediction generated using the trained "
            "model and saved scaler."
        )