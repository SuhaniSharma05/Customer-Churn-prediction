#[[gender", "SeniorCitizen", "Partner","Dependents", "PhoneService","PaperlessBilling", "tenure", "MonthlyCharges"]]
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(page_title="📉 Churn Prediction App", layout="wide")
st.title("📉 Customer Churn Prediction App")
st.subheader("💬 Predict whether a customer will churn")

# Tabs for layout
tab1, tab2 = st.tabs(["📋 Customer Input", "📊 Results Summary"])

with tab1:
    st.markdown("""
    ### 🧾 Instructions
    Fill in customer details below. Click *Predict* to see churn status.
    """)

    st.divider()
    st.success("✅ Input Section")

    # Input fields
    gender = st.selectbox("🧍 Gender", ["Male", "Female"])
    senior = st.selectbox("👵 Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("💍 Partner", ["No", "Yes"])
    dependents = st.selectbox("👶 Dependents", ["No", "Yes"])
    phone_service = st.selectbox("📞 Phone Service", ["Yes", "No"])
    paperless_billing = st.selectbox("📩 Paperless Billing", ["Yes", "No"])
    tenure = st.number_input("📆 Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charge = st.number_input("💸 Monthly Charge (₹)", min_value=30.0, max_value=200.0, value=75.0)

    st.divider()
    predict_button = st.button("🚀 Predict")

    if predict_button:
        # Encode inputs
        gender_val = 1 if gender == "Female" else 0
        senior_val = 1 if senior == "Yes" else 0
        partner_val = 1 if partner == "Yes" else 0
        dependents_val = 1 if dependents == "Yes" else 0
        phone_val = 1 if phone_service == "Yes" else 0
        billing_val = 1 if paperless_billing == "Yes" else 0

        # Feature list (match model training)
        features = [
            gender_val, senior_val, partner_val, dependents_val,
            phone_val, billing_val, tenure, monthly_charge
        ]

        input_df = pd.DataFrame([features], columns=[
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'tenure', 'MonthlyCharges'
        ])

        # Scale and predict
        x_scaled = scaler.transform(input_df)
        prediction = model.predict(x_scaled)[0]
        result = "❌ Yes (Customer will churn)" if prediction == 1 else "✅ No (Customer will stay)"

        # Store in session
        st.session_state["prediction_result"] = result
        st.session_state["input_df"] = input_df

with tab2:
    if "prediction_result" in st.session_state:
        st.subheader("🔍 Prediction Result")
        st.info(f"🎯 Churn Prediction: *{st.session_state['prediction_result']}*")

        churn_numeric = 1 if "Yes" in st.session_state["prediction_result"] else 0
        st.metric(label="📈 Churn Score", value=churn_numeric)

        st.write("### 🧾 Customer Summary")
        st.dataframe(st.session_state["input_df"])

        img = Image.open("image1.png")
        st.image(img, caption="Forecasting churn. Fueling retention.", use_container_width=True)
    else:
        st.warning("⚠ Make a prediction first in the 'Customer Input' tab")

# Footer
st.markdown("---")
st.caption("✨ Built by Suhani using Streamlit. Feel free to customize and enhance!")