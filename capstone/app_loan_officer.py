# app_loan_officer.py - Interactive Streamlit Web Interface
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Apex Credit Risk Portal", page_icon="🏦", layout="wide")

st.title("🏦 Apex Financial: Automated Loan Risk Assessment")
st.markdown("Enter applicant financial indicators below to evaluate credit default risk.")

# Sidebar Input Widgets
st.sidebar.header("Applicant Financial Profile")
age = st.sidebar.slider("Applicant Age", 18, 75, 34)
income = st.sidebar.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=75000, step=2500)
credit_score = st.sidebar.slider("Credit Bureau Score", 300, 850, 710)
loan_amt = st.sidebar.number_input("Requested Loan Amount ($)", min_value=1000, max_value=100000, value=25000, step=1000)
emp_years = st.sidebar.slider("Years in Current Employment", 0.0, 40.0, 6.0, step=0.5)
dti = st.sidebar.slider("Debt-to-Income (DTI) Ratio", 0.05, 0.80, 0.28, step=0.01)
home = st.sidebar.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN"])
purpose = st.sidebar.selectbox("Loan Purpose", ["DEBT_CONSOLIDATION", "HOME_IMPROVEMENT", "SMALL_BUSINESS", "MAJOR_PURCHASE"])

# Real-Time Decision Logic
applicant_record = pd.DataFrame([{
    'Age': age, 'AnnualIncome': income, 'CreditScore': credit_score,
    'LoanAmount': loan_amt, 'EmploymentYears': emp_years,
    'HomeOwnership': home, 'LoanPurpose': purpose, 'DTIRatio': dti
}])

# Simulated Model Probability Calculation
latent = -0.012 * credit_score + 3.2 * dti - 0.000025 * income + 0.00004 * loan_amt - 0.08 * emp_years + 3.8
risk_prob = float(1.0 / (1.0 + np.exp(-latent)))

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Default Probability", f"{risk_prob:.1%}")
col2.metric("Bureau Risk Rating", "TIER 1 (PRIME)" if credit_score >= 720 else "TIER 2 (NEAR-PRIME)" if credit_score >= 640 else "SUBPRIME")

with col3:
    if risk_prob < 0.25:
        st.success("✅ DECISION: APPROVED")
    elif risk_prob < 0.45:
        st.warning("⚠️ DECISION: MANUAL REVIEW REQUIRED")
    else:
        st.error("❌ DECISION: APPLICATION DECLINED")

st.markdown("---")
st.subheader("Key Driving Factors for This Applicant")
if credit_score < 620:
    st.info("• High Risk Driver: Bureau credit score is below institutional prime threshold.")
if dti > 0.40:
    st.info("• High Risk Driver: Debt-to-Income ratio exceeds recommended 40% ceiling.")
if income >= 70000:
    st.info("• Positive Compensating Factor: Robust annual income supports debt serviceability.")
