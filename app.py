import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import base64 # Required for file download link

# Set page configuration
st.set_page_config(page_title="Employee Attrition Batch Predictor", layout="wide")

# --- Load the trained model, scaler, and mappings ---
# Use st.cache_resource to load the large objects only once across reruns
@st.cache_resource
def load_artifacts():
    """Loads the model, scaler, and categorical mappings."""
    try:
        model = joblib.load('employee_attrition_model.pkl')
        scaler = joblib.load('scaler.pkl')
        categorical_mappings = joblib.load('categorical_mappings.pkl')
        # The scaler object should have `feature_names_in_` after fitting,
        # which gives the column order and names expected by the model input.
        feature_names = scaler.feature_names_in_
        # We also need the list of original categorical columns by name to apply mapping
        # These are the columns that were originally 'object' dtype and label encoded
        categorical_cols_original = list(categorical_mappings.keys())
        return model, scaler, categorical_mappings, feature_names, categorical_cols_original
    except FileNotFoundError:
        st.error("Error: Model, scaler, or mapping files not found.")
        st.error("Please ensure 'employee_attrition_model.pkl', 'scaler.pkl', and 'categorical_mappings.pkl' are in the same directory as this script.")
        st.stop() # Stop the app if files are not found
    except Exception as e:
        st.error(f"An error occurred while loading artifacts: {e}")
        st.stop()

model, scaler, categorical_mappings, feature_names, categorical_cols_original = load_artifacts()

# --- Helper function for preprocessing input batch data ---
def preprocess_input_batch(df: pd.DataFrame, scaler, categorical_mappings, feature_names, categorical_cols_original):
    """
    Applies preprocessing steps to a batch DataFrame read from a CSV.
    Expects DataFrame with original column names.
    """
    processed_df = df.copy()

    # Define columns to drop if they exist in the input file
    # These were dropped during training or are the target variable
    cols_to_drop_if_exist = ['EmployeeCount', 'StandardHours', 'Over18', 'MonthlyRate', 'HourlyRate', 'Attrition']
    cols_to_drop_in_input = [col for col in cols_to_drop_if_exist if col in processed_df.columns]
    processed_df = processed_df.drop(columns=cols_to_drop_in_input)

    # Apply label encoding to original categorical columns
    for col in categorical_cols_original:
        if col in processed_df.columns:
            # Map original string categories to numerical using loaded mappings
            # Handle potential unknown categories by mapping them to -1
            # This assumes the trained model (Random Forest) can handle -1 or a value outside trained range
            # For stricter handling, you might raise an error or map to a 'missing' category index if available
            unknown_categories = processed_df[col][~processed_df[col].isin(categorical_mappings[col].keys())].unique()
            if len(unknown_categories) > 0:
                 # Convert unknowns to strings for display in warning
                 unknown_str = ', '.join(map(str, unknown_categories))
                 st.warning(f"Warning: Unknown categories found in column '{col}': {unknown_str}. They will be mapped to -1.")

            processed_df[col] = processed_df[col].map(categorical_mappings[col]).fillna(-1).astype(int)
        # If a required original categorical column is missing, it will be handled in the next step
        # when we ensure all `feature_names` are present.


    # Ensure all required feature columns are present in the processed_df
    # and select/reorder them to match `feature_names`
    final_features_df = pd.DataFrame(index=processed_df.index)

    for feature in feature_names:
        if feature in processed_df.columns:
            # Feature exists (either originally numerical or now encoded categorical)
            final_features_df[feature] = processed_df[feature]
        else:
            # This happens if a required original numerical column is missing in the uploaded file,
            # or if an original categorical column was missing and its encoded name is now a feature name.
            st.error(f"Error: Required feature column '{feature}' not found in the uploaded file after initial processing.")
            st.stop() # Stop processing as we're missing a critical feature


    # Ensure dtypes are correct (usually float for scaler)
    final_features_df = final_features_df.astype(float)

    # Scale the features using the loaded scaler
    try:
        scaled_data = scaler.transform(final_features_df)
        scaled_df = pd.DataFrame(scaled_data, columns=feature_names, index=final_features_df.index)
    except Exception as e:
         st.error(f"Error during scaling: {e}")
         raise e # Re-raise to stop prediction

    return scaled_df, df # Return the scaled features and the original df (for adding results later)

# --- Streamlit App Interface ---
st.title("📦 Employee Attrition Batch Prediction")

st.write("""
Upload a CSV file containing employee data to get attrition predictions for the entire batch.
The app will add prediction results as new columns to your file.
""")

st.info("""
**File Requirements:**
- Must be a CSV file (`.csv`).
- Must contain columns corresponding to the features used in the model training.
- The column names should match the original dataset (e.g., 'Age', 'Gender', 'Department', 'MonthlyIncome', etc.).
- You do *not* need to include the 'Attrition' column if you are predicting on new data.
- Columns that were dropped during training ('EmployeeCount', 'StandardHours', 'Over18', 'MonthlyRate', 'HourlyRate') are not required.
""")

uploaded_file = st.file_uploader("Upload Employee Data CSV", type="csv")

if uploaded_file is not None:
    try:
        # Read the uploaded CSV
        original_input_df = pd.read_csv(uploaded_file)

        if original_input_df.empty:
            st.warning("The uploaded file is empty.")
            st.stop()

        st.write("Original Data Preview:")
        st.dataframe(original_input_df.head())

        # --- Preprocess the data ---
        st.write("Preprocessing data for prediction...")
        # Pass the original df to preprocessing
        scaled_input_df, df_for_results = preprocess_input_batch(
            original_input_df,
            scaler,
            categorical_mappings,
            feature_names,
            categorical_cols_original
        )
        st.write("Preprocessing complete.")


        # --- Make Predictions ---
        st.write("Making predictions...")
        predictions = model.predict(scaled_input_df)
        probabilities = model.predict_proba(scaled_input_df)

        # --- Add Predictions to DataFrame for Results ---
        df_for_results['Predicted_Attrition'] = predictions
        # Map numerical predictions back to 'Yes'/'No' for clarity
        df_for_results['Predicted_Attrition'] = df_for_results['Predicted_Attrition'].map({0: 'No', 1: 'Yes'})
        df_for_results['Attrition_Probability_Yes'] = probabilities[:, 1]
        df_for_results['Attrition_Probability_No'] = probabilities[:, 0]

        st.write("Predictions complete.")

        # --- Display Results ---
        st.subheader("Prediction Results (first 100 rows):")
        st.dataframe(df_for_results.head(100)) # Show only head for potentially large files


        # --- Download Link ---
        # Function to convert DataFrame to CSV for download
        @st.cache_data # Cache the CSV conversion result
        def convert_df_to_csv(df):
            # IMPORTANT: Ensure the file is encoded correctly for download
            return df.to_csv(index=False).encode('utf-8')

        csv_data = convert_df_to_csv(df_for_results)

        st.download_button(
            label="⬇️ Download Full Results as CSV",
            data=csv_data,
            file_name='employee_attrition_predictions.csv',
            mime='text/csv',
            help="Download the original data with added prediction columns ('Predicted_Attrition', 'Attrition_Probability_Yes', 'Attrition_Probability_No')."
        )
        st.info("The full prediction results are available for download.")


    except pd.errors.EmptyDataError:
        st.error("Error: The uploaded file is empty.")
    except pd.errors.ParserError:
        st.error("Error: Could not parse the CSV file. Please ensure it's a valid CSV format.")
    except KeyError as e:
         st.error(f"Error: A required column is missing or misnamed in the uploaded file after initial processing steps: {e}.")
         st.info(f"The model expects features based on the following columns (after encoding categorical ones): {list(feature_names)}. Please ensure your input file contains the original columns needed to derive these.")
         st.write("Example of required original columns: Age, Gender, Department, MonthlyIncome, JobRole, etc.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.write("Please check the file format and column names against the requirements.")

# --- Sidebar ---
st.sidebar.header("About This App")
st.sidebar.info(
    """
    This app provides a batch prediction interface for employee attrition.
    Upload a CSV, and the app will add predictions using the trained model.
    """
)
st.sidebar.header("Required Files")
st.sidebar.warning("""
Ensure the following files are in the same directory as 'app_batch.py':
- `employee_attrition_model.pkl`
- `scaler.pkl`
- `categorical_mappings.pkl`
""")
st.sidebar.markdown("---")
st.sidebar.write("Created with Streamlit")
