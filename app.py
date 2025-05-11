import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import base64 # Required for file download link

# Set page configuration
st.set_page_config(page_title="Project2_Group17 | Employee Attrition Batch Predictor", layout="wide")

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
        # This is robust even if columns were dropped or reordered internally during training preprocessing
        feature_names = scaler.feature_names_in_
        # We also need the list of original categorical columns by name to apply mapping
        # These are the columns that were originally 'object' dtype and label encoded
        # We get these from the keys of the saved categorical_mappings dictionary
        categorical_cols_original = list(categorical_mappings.keys())
        return model, scaler, categorical_mappings, feature_names, categorical_cols_original
    except FileNotFoundError:
        st.error("Error: Model, scaler, or mapping files not found.")
        st.error("Please ensure 'employee_attrition_model.pkl', 'scaler.pkl', and 'categorical_mappings.pkl' are in the same directory as this script.")
        st.stop() # Stop the app if files are not found
    except Exception as e:
        st.error(f"An error occurred while loading artifacts: {e}")
        st.stop() # Stop the app on other loading errors

# Load artifacts at the start
model, scaler, categorical_mappings, feature_names, categorical_cols_original = load_artifacts()

# --- Helper function for preprocessing input batch data ---
def preprocess_input_batch(df: pd.DataFrame, scaler, categorical_mappings, feature_names, categorical_cols_original):
    """
    Applies preprocessing steps to a batch DataFrame read from a CSV.
    Expects DataFrame with original column names as per the training data.
    Ensures the output DataFrame has columns matching feature_names in order.
    """
    processed_df = df.copy() # Work on a copy

    # Define columns to drop if they exist in the input file
    # These were dropped during training or are the target variable
    # Ensure these match the columns dropped in the training script
    cols_to_drop_if_exist = [
        'EmployeeCount',
        'StandardHours',
        'Over18',
        'MonthlyRate', # Dropped in training script
        'HourlyRate',  # Dropped in training script
        'Attrition'    # Target variable, not a feature
    ]
    cols_to_drop_in_input = [col for col in cols_to_drop_if_exist if col in processed_df.columns]
    if cols_to_drop_in_input:
        processed_df = processed_df.drop(columns=cols_to_drop_in_input)
        # st.info(f"Dropped columns from input: {', '.join(cols_to_drop_in_input)}") # Optional: inform user

    # Apply label encoding to original categorical columns
    for col in categorical_cols_original:
        if col in processed_df.columns:
            # Map original string categories to numerical using loaded mappings
            # Handle potential unknown categories by mapping them to -1
            # This strategy assumes the model (Random Forest) can handle -1 values gracefully.
            # For stricter handling, you might raise an error or map to a 'missing' category index if trained to do so.
            unknown_categories = processed_df[col][~processed_df[col].astype(str).isin(categorical_mappings[col].keys())].unique()
            if len(unknown_categories) > 0:
                # Convert unknowns to strings for display in warning
                unknown_str = ', '.join(map(str, unknown_categories))
                st.warning(f"Warning: Unknown categories found in column '{col}': {unknown_str}. They will be mapped to a default value (-1) or the nearest known mapping.")

            # Convert column to string type first to handle potential mixed types gracefully before mapping
            # Then apply mapping, fill NaNs (resulting from unknown categories or original NaNs) with -1, and convert to int
            processed_df[col] = processed_df[col].astype(str).map(categorical_mappings[col]).fillna(-1).astype(int)
        else:
             # If an original categorical column expected by the model is missing in the input file,
             # we cannot proceed. This will be caught later when checking feature_names,
             # but an explicit warning/error here could also be useful depending on strictness required.
             pass # Let the feature_names check handle completely missing required columns

    # Ensure all required feature columns are present and in the correct order
    # required by the scaler/model. The `feature_names` loaded from the scaler
    # represent the exact column names and order expected by the model input.
    final_features_df = pd.DataFrame(index=processed_df.index)

    for feature in feature_names:
        if feature in processed_df.columns:
            # Feature exists (either originally numerical or now encoded categorical)
            final_features_df[feature] = processed_df[feature]
        else:
            # This happens if a required original numerical column is missing in the uploaded file,
            # or if an original categorical column was missing and its encoded name is now a feature name.
            # This indicates a major discrepancy between expected input and the provided file.
            st.error(f"Error: Required feature column '{feature}' not found in the processed file. Please ensure your input CSV contains all necessary original columns.")
            # Display columns found in the input file for user debugging
            st.info(f"Columns found in uploaded file (after initial drops): {list(processed_df.columns)}")
            st.info(f"Expected final feature columns (from model training): {list(feature_names)}")
            st.stop() # Stop processing as we're missing a critical feature


    # Ensure dtypes are correct (usually float for scaler)
    # Use errors='coerce' to turn non-numeric values into NaN, though the above steps should prevent this.
    # Fill any potential remaining NaNs with 0 or a strategy appropriate for the model/features.
    # StandardScaler expects float input and can handle NaNs depending on its config, but 0 is a simple default filler.
    # However, the scaler was likely fit on data without NaNs (post-SMOTE, post-encoding), so NaNs here would indicate a major input data issue.
    # Let's assume input should be clean numeric/encoded values at this point.
    try:
         final_features_df = final_features_df.astype(float)
         # Simple check for NaNs that might have slipped through
         if final_features_df.isnull().any().any():
             st.error("Error: NaN values found in the prepared features before scaling. Please check your input data for missing or unhandleable values.")
             st.dataframe(final_features_df[final_features_df.isnull().any(axis=1)].head()) # Show rows with NaNs
             st.stop()

    except ValueError as e:
        st.error(f"Error converting features to numeric type: {e}. Please check column data types in your input file.")
        st.dataframe(final_features_df.head()) # Show problematic data
        st.stop()


    # Scale the features using the loaded scaler
    try:
        scaled_data = scaler.transform(final_features_df)
        scaled_df = pd.DataFrame(scaled_data, columns=feature_names, index=final_features_df.index)
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        st.error("Please ensure the data types and ranges of your input match the data used for training the scaler.")
        raise e # Re-raise to stop prediction

    # Return the scaled features for the model and the original df (for adding results later)
    return scaled_df, df # Returning the original df before preprocessing steps is better


# --- Streamlit App Interface ---
st.title("Project2_Group17\n📦 Employee Attrition Batch Prediction")

st.write("""
Upload a CSV file containing employee data to get attrition predictions for the entire batch.
The app will add prediction results as new columns to your file.
""")
st.info("""
**Testing Dummy Data in Github Repo "DummyTestingData.csv"**

""")
st.info("""
**File Requirements:**
- Must be a CSV file (`.csv`).
- Must contain columns corresponding to the features used during model training (e.g., Age, Gender, Department, MonthlyIncome, JobRole, etc.).
- The following columns will be ignored if present: `EmployeeCount`, `StandardHours`, `Over18`, `MonthlyRate`, `HourlyRate`, `Attrition`.
- For internal use only - HR department.
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
        # The function returns the scaled features and the ORIGINAL df for results
        scaled_input_df, df_for_results = preprocess_input_batch(
            original_input_df,
            scaler,
            categorical_mappings,
            feature_names,
            categorical_cols_original
        )
        st.write("Preprocessing complete. Ready to predict.")


        # --- Make Predictions ---
        st.write("Making predictions...")
        predictions = model.predict(scaled_input_df)
        probabilities = model.predict_proba(scaled_input_df)

        # --- Add Predictions to DataFrame for Results ---
        # Ensure the predictions align with the original indices
        df_for_results['Predicted_Attrition'] = predictions
        # Map numerical predictions back to 'Yes'/'No' for clarity
        # Assumes 0 was 'No' and 1 was 'Yes' during training
        # You could load the le_attrition encoder to be more robust if you saved it
        df_for_results['Predicted_Attrition'] = df_for_results['Predicted_Attrition'].map({0: 'No', 1: 'Yes'})
        df_for_results['Attrition_Probability_No'] = probabilities[:, 0] # Probability of class 0 ('No')
        df_for_results['Attrition_Probability_Yes'] = probabilities[:, 1] # Probability of class 1 ('Yes')


        st.write("Predictions complete.")

        # --- Display Results ---
        st.subheader("Prediction Results (first 100 rows):")
        # Show the original columns plus the new prediction columns
        st.dataframe(df_for_results.head(100))


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
            help="Download the original data with added prediction columns ('Predicted_Attrition', 'Attrition_Probability_No', 'Attrition_Probability_Yes')."
        )
        st.info("The full prediction results are available for download.")


    except pd.errors.EmptyDataError:
        st.error("Error: The uploaded file is empty.")
    except pd.errors.ParserError:
        st.error("Error: Could not parse the CSV file. Please ensure it's a valid CSV format.")
    except KeyError as e:
         st.error(f"Error processing columns: A required column might be missing or misnamed based on the trained model: {e}.")
         st.info(f"The model expects features based on the following columns (after encoding categorical ones and dropping others): {list(feature_names)}. Please ensure your input file contains the original columns needed to derive these.")
         st.write("Example of required original columns: Age, Gender, Department, MonthlyIncome, JobRole, etc.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.write("Please check the file format and column names against the requirements and ensure the model artifacts are correctly loaded.")

# --- Sidebar ---
st.sidebar.header("About This App")
st.sidebar.info(
    """
    This app provides a batch prediction interface for employee attrition using a trained Random Forest model.
    Upload a CSV file containing employee data, and the app will add predictions to your file.

    The model was trained on historical data and uses features like Age, Department, Job Role, Income, etc.
    It predicts the likelihood of an employee leaving the company.
    """
)
st.sidebar.markdown("---")
st.sidebar.write("Created by Project2_Group17 with Streamlit")
