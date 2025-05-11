import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Load the saved models and encoders
# =============================================================================
model = joblib.load('employee_attrition_model.pkl')
scaler = joblib.load('scaler.pkl')
categorical_mappings = joblib.load('categorical_mappings.pkl')

# =============================================================================
# Streamlit App Configuration
# =============================================================================
st.title('Employee Attrition Prediction App')
st.write('Predict the likelihood of an employee leaving the company based on various features.')

# =============================================================================
# User Inputs
# =============================================================================
def user_input_features():
    age = st.number_input('Age', min_value=18, max_value=65, value=30)
    daily_rate = st.number_input('Daily Rate', min_value=100, max_value=1500, value=800)
    distance_from_home = st.number_input('Distance from Home', min_value=1, max_value=30, value=5)
    education = st.selectbox('Education Level', [1, 2, 3, 4, 5])
    environment_satisfaction = st.selectbox('Environment Satisfaction', [1, 2, 3, 4])
    job_satisfaction = st.selectbox('Job Satisfaction', [1, 2, 3, 4])
    marital_status = st.selectbox('Marital Status', list(categorical_mappings['MaritalStatus'].keys()))
    monthly_income = st.number_input('Monthly Income', min_value=1000, max_value=20000, value=5000)
    num_companies_worked = st.number_input('Number of Companies Worked', min_value=0, max_value=10, value=1)
    over_time = st.selectbox('OverTime', list(categorical_mappings['OverTime'].keys()))
    total_working_years = st.number_input('Total Working Years', min_value=0, max_value=40, value=10)
    years_at_company = st.number_input('Years at Company', min_value=0, max_value=40, value=5)
    years_since_last_promotion = st.number_input('Years Since Last Promotion', min_value=0, max_value=15, value=1)
    years_with_curr_manager = st.number_input('Years with Current Manager', min_value=0, max_value=15, value=3)

    data = {
        'Age': age,
        'DailyRate': daily_rate,
        'DistanceFromHome': distance_from_home,
        'Education': education,
        'EnvironmentSatisfaction': environment_satisfaction,
        'JobSatisfaction': job_satisfaction,
        'MaritalStatus': categorical_mappings['MaritalStatus'][marital_status],
        'MonthlyIncome': monthly_income,
        'NumCompaniesWorked': num_companies_worked,
        'OverTime': categorical_mappings['OverTime'][over_time],
        'TotalWorkingYears': total_working_years,
        'YearsAtCompany': years_at_company,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# =============================================================================
# Model Prediction
# =============================================================================
st.subheader('Prediction Result')
if st.button('Predict'):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.error(f'The employee is likely to leave the company with a probability of {prediction_proba[0][1]:.2f}')
    else:
        st.success(f'The employee is likely to stay in the company with a probability of {prediction_proba[0][0]:.2f}')
