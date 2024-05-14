import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

st.title('Health Claim Fraud Predictor')

@st.cache_data
def preprocess_data(df):
    # Drop unnecessary columns
    DropCols = ['index', 'National Provider Identifier',
                'Last Name/Organization Name of the Provider',
                'First Name of the Provider', 'Middle Initial of the Provider',
                'Street Address 1 of the Provider', 'Street Address 2 of the Provider',
                'City of the Provider', 'Country Code of the Provider',
                'Zip Code of the Provider', "HCPCS Code", "HCPCS Description"]
    df = df.drop(DropCols, axis=1)

    # Replace variations of 'Credentials of the Provider'
    df['Credentials of the Provider'] = df['Credentials of the Provider'].replace({
        'M.D': 'MD', 'MD.': 'MD', 'M.D.': 'MD', 'D.O.': 'DO'
    })

    # Identify columns with missing values
    columns_with_missing_values = df.columns[df.isnull().any()]

    # Create an instance of SimpleImputer with strategy='most_frequent'
    imputer = SimpleImputer(strategy='most_frequent')

    # Fit the imputer on the DataFrame to learn the mode of each column with missing values
    imputer.fit(df[columns_with_missing_values])

    # Transform the DataFrame to replace missing values with the learned modes
    df[columns_with_missing_values] = imputer.transform(df[columns_with_missing_values])

    def RemoveComma(x):
        if isinstance(x, str):  # Check if the data type is a string
            return x.replace(",", "")
        else:
            return x

    # Remove commas and convert to numeric
    numeric_columns = ["Average Medicare Allowed Amount", "Average Submitted Charge Amount",
                       "Average Medicare Payment Amount", "Average Medicare Standardized Amount",
                       "Number of Services", "Number of Medicare Beneficiaries",
                       "Number of Distinct Medicare Beneficiary/Per Day Services"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column].apply(RemoveComma), errors="coerce")

    return df

@st.cache_data
def load_data():
    # Read the CSV file
    url = 'https://raw.githubusercontent.com/cmrajendran/healthcare_claim_fraud_detection/main/Healthcare%20Providers.csv'
    df = pd.read_csv(url)

    # Preprocess the data
    df = preprocess_data(df)

    return df


def main():
    # Load the data
    df = load_data()

    # Input variables
    credentials = st.text_input("Credentials of the Provider")
    gender = st.selectbox("Gender of the Provider", df['Gender of the Provider'].unique())
    entity_type_provider = st.selectbox("Entity Type of the Provider", df['Entity Type of the Provider'].unique())
    state_code_provider = st.selectbox("State Code of the Provider", df['State Code of the Provider'].unique())
    provider_type = st.selectbox("Provider Type", df['Provider Type'].unique())
    medicare_participation_indicator = st.selectbox("Medicare Participation Indicator",
                                                    df['Medicare Participation Indicator'].unique())
    place_of_service = st.selectbox("Place of Service", df['Place of Service'].unique())
    hcpcs_drug_indicator = st.selectbox("HCPCS Drug Indicator", df['HCPCS Drug Indicator'].unique())

    # Numeric input variables
    number_of_services = st.number_input("Number of Services", min_value=0.0)
    number_of_medicare_beneficiaries = st.number_input("Number of Medicare Beneficiaries", min_value=0)
    number_of_distinct_medicare_beneficiary_per_day_services = st.number_input(
        "Number of Distinct Medicare Beneficiary/Per Day Services", min_value=0)
    average_medicare_allowed_amount = st.number_input("Average Medicare Allowed Amount", min_value=0.0)
    average_submitted_charge_amount = st.number_input("Average Submitted Charge Amount", min_value=0.0)
    average_medicare_payment_amount = st.number_input("Average Medicare Payment Amount", min_value=0.0)
    average_medicare_standardized_amount = st.number_input("Average Medicare Standardized Amount", min_value=0.0)

    # Identify categorical columns to be encoded
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Perform one-hot encoding on the categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns)

    # Split the data into training and testing sets
    X_train, X_test = train_test_split(df_encoded, test_size=0.2, random_state=42)

    # Train Isolation Forest model on the original training data
    iforest = IsolationForest(contamination=0.05)
    iforest.fit(X_train)

    # Predict anomalies in the testing data
    anomalies = iforest.predict(X_test)


    # Display model prediction
    st.write("Anomalies detected in testing data:", anomalies)

    # Input submission
    if st.button("Submit"):
        # Create a DataFrame for user input
        user_input_df = pd.DataFrame({
            'Number of Services': [number_of_services],
            'Number of Medicare Beneficiaries': [number_of_medicare_beneficiaries],
            'Number of Distinct Medicare Beneficiary/Per Day Services': [
                number_of_distinct_medicare_beneficiary_per_day_services],
            'Average Medicare Allowed Amount': [average_medicare_allowed_amount],
            'Average Submitted Charge Amount': [average_submitted_charge_amount],
            'Average Medicare Payment Amount': [average_medicare_payment_amount],
            'Average Medicare Standardized Amount': [average_medicare_standardized_amount],
            # Add other features if available
        })

        # Ensure user input has the same features as training data
        for col in categorical_columns:
            user_input_df[col] = 0

        # Perform one-hot encoding on the user input DataFrame
        user_input_encoded = pd.get_dummies(user_input_df, columns=categorical_columns)

        # Ensure user input DataFrame has the same columns as training data
        user_input_encoded = user_input_encoded.reindex(columns=X_train.columns, fill_value=0)

        # Predict anomaly for user input
        user_anomaly = iforest.predict(user_input_encoded)
        # Label 1 (normal samples) as 0 and -1 (anomalous samples) as 1 in the prediction results
        user_anomaly[user_anomaly == 1] = 0
        user_anomaly[user_anomaly == -1] = 1

        # Interpret prediction results
        if user_anomaly == 0:
            st.write("Prediction: No Suspected Fraud")
        else:
            st.write("Prediction: Suspected Fraud. Please investigate.")


if __name__ == "__main__":
    main()
