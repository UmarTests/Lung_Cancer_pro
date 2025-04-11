import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load(r"C:\Users\mohdq\OneDrive\Desktop\internship projects\Lung_Cancer.pro\Lung Cancer\lung_cancer_model.pkl")

st.title('Lung Cancer Survival Predictor')

# Input widgets
col1, col2 = st.columns(2)
with col1:
    age = st.slider('Age', 0, 100, 30)
    bmi = st.slider('BMI', 15, 40, 25)
    cholesterol = st.slider('Cholesterol Level', 50, 300, 100)
with col2:
    treatment = st.selectbox('Treatment', ['Surgery', 'Chemotherapy', 'Radiation', 'Combined'])
    stage = st.selectbox('Cancer Stage', ['Stage I', 'Stage II', 'Stage III', 'Stage IV'])
    health_risks = st.slider('Number of Health Risk Factors', 0, 4, 1)

# Convert inputs to model features
treatment_scores = {
    'Surgery': 1.0,
    'Combined': 0.7,
    'Chemotherapy': 0.5,
    'Radiation': 0.3
}
stage_scores = {
    'Stage I': 1.0,
    'Stage II': 1.5,
    'Stage III': 2.0,
    'Stage IV': 2.5
}

if st.button('Predict Survival'):
    # Create all required features
    input_data = pd.DataFrame([[
        age,
        bmi,
        cholesterol,
        health_risks,
        treatment_scores[treatment],
        stage_scores[stage],
        bmi * cholesterol,  # bmi_cholesterol_interaction
        age * health_risks  # age_health_risk
    ]], columns=[
        'age', 'bmi', 'cholesterol_level', 'health_risk_factors',
        'treatment_score', 'stage_score', 
        'bmi_cholesterol_interaction', 'age_health_risk'
    ])
    
    # Get prediction
    proba = model.predict_proba(input_data)[0][1]
    
    # Convert probability to 3.5-6.5 scale
    survival_score = 3.5 + (proba * 3)  # Maps 0% → 3.5, 100% → 6.5
    
    # Determine prediction (using threshold 0.5 unless changed in model)
    prediction = model.predict(input_data)[0]
    
    # Display results
    st.subheader("Results")
    
    # Show survival score with color indication
    score_color = "green" if survival_score >= 5.0 else "orange" if survival_score >= 4.25 else "red"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Survival Probability", f"{proba:.0%}")
    with col2:
        st.markdown(f"<h3 style='text-align: center; color: {score_color};'>Survival Score: {survival_score:.1f}</h3>", 
                   unsafe_allow_html=True)
    with col3:
        st.metric("Prediction", "Survive" if prediction else "Not Survive")
    
    # Interpretation guide
    st.markdown("""
    **Score Interpretation:**
    - 🔴 3.5-4.2: High risk
    - 🟠 4.2-5.0: Moderate risk 
    - 🟢 5.0-6.5: Low risk
    """)
    
    # Show input features (optional)
    st.expander("Show generated features").write(input_data)