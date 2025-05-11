import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Set page configuration
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

# --- Load the trained model, scaler, and mappings ---
# Use st.cache_resource to load the large objects only once
@st.cache_resource
def load_artifacts():
    """Loads the model, scaler, and categorical mappings."""
    try:
        model = joblib.load('employee_attrition_model.pkl')
        scaler = joblib.load('scaler.pkl')
        categorical_mappings = joblib.load('categorical_mappings.pkl')
        # The scaler object should have `feature_names_in_` after fitting,
        # which gives the column order expected by the model.
        feature_names = scaler.feature_names_in_
        return model, scaler, categorical_mappings, feature_names
    except FileNotFoundError:
        st.error("Error: Model, scaler, or mapping files not found.")
        st.error("Please ensure 'employee_attrition_model.pkl', 'scaler.pkl', and 'categorical_mappings.pkl' are in the same directory.")
        st.stop() # Stop the app if files are not found
    except Exception as e:
        st.error(f"An error occurred while loading artifacts: {e}")
        st.stop()

model, scaler, categorical_mappings, feature_names = load_artifacts()

# Get original column names for categorical features from mappings
# These are the columns that were originally 'object' dtype and were label encoded
categorical_cols_encoded = list(categorical_mappings.keys())

# --- Streamlit App Interface ---
st.title("🔮 Employee Attrition Prediction App")

st.write("""
This application uses a trained machine learning model to predict whether an employee is likely to leave the company based on their characteristics.
Please enter the employee's details below:
""")

# --- Input Form Layout ---
# Use columns to organize the input fields
col1, col2, col3 = st.columns(3)

# Dictionary to store user inputs
input_data = {}

# --- Input Fields ---
# Group inputs logically
with col1:
    st.subheader("Personal & Job Details")
    input_data['Age'] = st.slider('Age', min_value=18, max_value=67, value=35, help="Employee's current age.")
    input_data['Gender'] = st.selectbox('Gender', options=list(categorical_mappings['Gender'].keys()))
    input_data['MaritalStatus'] = st.selectbox('Marital Status', options=list(categorical_mappings['MaritalStatus'].keys()))
    input_data['EmployeeNumber'] = st.number_input('Employee Number (ID)', min_value=1, value=1001, help="Unique employee identifier (included as a feature in the model).")
    input_data['Department'] = st.selectbox('Department', options=list(categorical_mappings['Department'].keys()))
    input_data['JobRole'] = st.selectbox('Job Role', options=list(categorical_mappings['JobRole'].keys()))
    input_data['JobLevel'] = st.selectbox('Job Level', options=[1, 2, 3, 4, 5], help="Employee's job level (1 to 5).")
    input_data['OverTime'] = st.selectbox('Over Time', options=list(categorical_mappings['OverTime'].keys()))


with col2:
    st.subheader("Work & Environment Factors")
    input_data['DistanceFromHome'] = st.number_input('Distance From Home (miles)', min_value=1, max_value=29, value=10, help="Distance from home to work in miles.")
    input_data['NumCompaniesWorked'] = st.number_input('Number of Companies Worked', min_value=0, max_value=9, value=1, help="Total number of companies the employee has worked for.")
    input_data['TotalWorkingYears'] = st.number_input('Total Working Years', min_value=0, max_value=40, value=10, help="Total years the employee has worked throughout their career.")
    input_data['YearsAtCompany'] = st.number_input('Years at Company', min_value=0, max_value=40, value=5, help="Total number of years the employee has been with the current company.")
    input_data['YearsInCurrentRole'] = st.number_input('Years in Current Role', min_value=0, max_value=18, value=3, help="Number of years spent in the current job role at the company.")
    input_data['YearsSinceLastPromotion'] = st.number_input('Years Since Last Promotion', min_value=0, max_value=15, value=1, help="Number of years since the employee's last promotion.")
    input_data['YearsWithCurrManager'] = st.number_input('Years With Current Manager', min_value=0, max_value=17, value=3, help="Number of years reporting to the current manager.")
    input_data['EnvironmentSatisfaction'] = st.selectbox('Environment Satisfaction', options=[1, 2, 3, 4], format_func=lambda x: f'{x}: {["Low", "Good", "Great", "Excellent"][x-1]}', help="Satisfaction with the work environment (1-4).")
    input_data['JobInvolvement'] = st.selectbox('Job Involvement', options=[1, 2, 3, 4], format_func=lambda x: f'{x}: {["Low", "Medium", "High", "Very High"][x-1]}', help="Level of job involvement (1-4).")


with col3:
    st.subheader("Compensation & Satisfaction")
    input_data['DailyRate'] = st.number_input('Daily Rate', min_value=102, max_value=1499, value=800, help="Employee's daily pay rate.")
    input_data['MonthlyIncome'] = st.number_input('Monthly Income', min_value=1009, max_value=19999, value=5000, help="Employee's monthly income.")
    input_data['PercentSalaryHike'] = st.number_input('Percent Salary Hike (%)', min_value=11, max_value=25, value=12, help="Percentage increase in salary last year.")
    input_data['StockOptionLevel'] = st.selectbox('Stock Option Level', options=[0, 1, 2, 3], help="Level of stock options held by the employee (0-3).")
    input_data['Education'] = st.selectbox('Education', options=[1, 2, 3, 4, 5], format_func=lambda x: f'{x}: {["Below College", "College", "Bachelor", "Master", "Doctor"][x-1]}', help="Education level (1-5).")
    input_data['EducationField'] = st.selectbox('Education Field', options=list(categorical_mappings['EducationField'].keys()))
    input_data['JobSatisfaction'] = st.selectbox('Job Satisfaction', options=[1, 2, 3, 4], format_func=lambda x: f'{x}: {["Low", "Good", "Great", "Excellent"][x-1]}', help="Satisfaction with the job itself (1-4).")
    input_data['PerformanceRating'] = st.selectbox('Performance Rating', options=[1, 2, 3, 4], format_func=lambda x: f'{x}: {["Low", "Good", "Excellent", "Outstanding"][x-1]}', help="Employee's performance rating (usually 3 or 4 in the dataset).")
    input_data['RelationshipSatisfaction'] = st.selectbox('Relationship Satisfaction', options=[1, 2, 3, 4], format_func=lambda x: f'{x}: {["Low", "Good", "Great", "Excellent"][x-1]}', help="Satisfaction with relationships at work (1-4).")
    input_data['TrainingTimesLastYear'] = st.number_input('Training Times Last Year', min_value=0, max_value=6, value=2, help="Number of times training was conducted last year.")
    input_data['WorkLifeBalance'] = st.selectbox('Work Life Balance', options=[1, 2, 3, 4], format_func=lambda x: f'{x}: {["Bad", "Good", "Better", "Best"][x-1]}', help="Work-life balance rating (1-4).")


# --- Prediction Button ---
st.markdown("---") # Add a separator
if st.button('Predict Attrition', help="Click to get the attrition prediction."):
    # Create a DataFrame from the input data
    # Ensure it's a single row DataFrame
    input_df = pd.DataFrame([input_data])

    # --- Preprocessing the input data ---
    # Apply the same label encoding as done during training
    for col in categorical_cols_encoded:
        if col in input_df.columns:
            # Use the loaded mapping dictionary
            input_df[col] = input_df[col].map(categorical_mappings[col])
            # Handle potential missing values if a category wasn't in training data (shouldn't happen with selectbox)
            # input_df[col] = input_df[col].fillna(-1) # Example fillna strategy

    # Ensure the columns are in the exact order the model was trained on
    # This is crucial for correct prediction
    try:
        # Use the feature_names list obtained from the loaded scaler
        input_df = input_df[feature_names]
    except KeyError as e:
        st.error(f"Error: Missing feature in input data or incorrect column name: {e}")
        st.error("Please ensure all input fields are filled correctly.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during feature reordering: {e}")
        st.stop()


    # Scale the numerical features using the loaded scaler
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
         st.error(f"An error occurred during scaling: {e}")
         st.stop()

    # --- Make Prediction ---
    try:
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # --- Display Prediction ---
        st.subheader("Prediction Result:")

        # The model was trained with Attrition: No=0, Yes=1
        if prediction[0] == 1:
            st.markdown("<h3 style='color: red;'>Predicted Attrition: Yes 😥</h3>", unsafe_allow_html=True)
            st.write(f"Confidence (Probability of Attrition): **{prediction_proba[0][1]:.2%}**")
            st.write(f"Confidence (Probability of No Attrition): {prediction_proba[0][0]:.2%}")
        else:
            st.markdown("<h3 style='color: green;'>Predicted Attrition: No 😊</h3>", unsafe_allow_html=True)
            st.write(f"Confidence (Probability of No Attrition): **{prediction_proba[0][0]:.2%}**")
            st.write(f"Confidence (Probability of Attrition): {prediction_proba[0][1]:.2%}")

        st.write("*(Prediction based on the trained Random Forest model)*")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Sidebar ---
st.sidebar.header("How to Use")
st.sidebar.info(
    """
    1. Make sure 'employee_attrition_model.pkl', 'scaler.pkl',
       and 'categorical_mappings.pkl' are in the same directory
       as this 'app.py' file.
    2. Fill out the details of the employee in the input fields.
    3. Click the 'Predict Attrition' button.
    4. The prediction result and confidence will be displayed below.
    """
)
st.sidebar.header("Model Details")
st.sidebar.write("This app uses a Random Forest Classifier trained on the HR Employee Attrition dataset.")
st.sidebar.write("SMOTE was applied to handle class imbalance.")
st.sidebar.write("Features were scaled using StandardScaler.")

st.sidebar.markdown("---")
st.sidebar.write("Created with Streamlit")
