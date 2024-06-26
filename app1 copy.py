import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Set the app title and description
st.set_page_config(page_title='Health Claim Fraud Predictor', page_icon=":hospital:", layout="centered")
st.title('Health Claim Fraud Predictor')
st.markdown("""
Welcome to the Health Claim Fraud Predictor! This tool helps in identifying potential fraudulent healthcare claims based on provider and claim information.
Please fill out the details below to get started.
""")

# Add a relevant title image
st.image("https://www.canva.com/design/DAGF8xxNsYk/EYfJ1V7m5IRKO0wVuB1aFg/view?utm_content=DAGF8xxNsYk&utm_campaign=designshare&utm_medium=link&utm_source=editor", use_column_width=True)

@st.cache_data
def preprocess_data(df):
    DropCols = ['index', 'National Provider Identifier',
                'Last Name/Organization Name of the Provider',
                'First Name of the Provider', 'Middle Initial of the Provider',
                'Street Address 1 of the Provider', 'Street Address 2 of the Provider',
                'City of the Provider', 'Country Code of the Provider',
                'Zip Code of the Provider', "HCPCS Code", "HCPCS Description"]
    df = df.drop(DropCols, axis=1)

    df['Credentials of the Provider'] = df['Credentials of the Provider'].replace({
        'M.D': 'MD', 'MD.': 'MD', 'M.D.': 'MD', 'D.O.': 'DO'
    })

    columns_with_missing_values = df.columns[df.isnull().any()]
    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(df[columns_with_missing_values])
    df[columns_with_missing_values] = imputer.transform(df[columns_with_missing_values])

    def RemoveComma(x):
        if isinstance(x, str):
            return x.replace(",", "")
        else:
            return x

    numeric_columns = ["Average Medicare Allowed Amount", "Average Submitted Charge Amount",
                       "Average Medicare Payment Amount", "Average Medicare Standardized Amount",
                       "Number of Services", "Number of Medicare Beneficiaries",
                       "Number of Distinct Medicare Beneficiary/Per Day Services"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column].apply(RemoveComma), errors="coerce")

    return df

@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/cmrajendran/healthcare_claim_fraud_detection/main/Healthcare%20Providers.csv'
    df = pd.read_csv(url)
    df = df.sample(frac=0.3, random_state=42)
    df = preprocess_data(df)
    return df

def main():
    df = load_data()

    st.sidebar.header("Provider Information")

    credentials = st.sidebar.text_input("Credentials of the Provider", help="Enter the credentials of the provider (e.g., MD, DO).")
    gender = st.sidebar.selectbox("Gender of the Provider", df['Gender of the Provider'].unique(), help="Select the gender of the provider.")
    entity_type_provider = st.sidebar.selectbox("Entity Type of the Provider", df['Entity Type of the Provider'].unique(), help="Select the entity type of the provider.")
    state_code_provider = st.sidebar.selectbox("State Code of the Provider", df['State Code of the Provider'].unique(), help="Select the state code of the provider.")
    provider_type = st.sidebar.selectbox("Provider Type", df['Provider Type'].unique(), help="Select the type of provider.")
    medicare_participation_indicator = st.sidebar.selectbox("Medicare Participation Indicator", df['Medicare Participation Indicator'].unique(), help="Select the Medicare participation indicator.")
    place_of_service = st.sidebar.selectbox("Place of Service", df['Place of Service'].unique(), help="Select the place of service.")
    hcpcs_drug_indicator = st.sidebar.selectbox("HCPCS Drug Indicator", df['HCPCS Drug Indicator'].unique(), help="Select the HCPCS drug indicator.")

    st.sidebar.header("Claim Information")

    number_of_services = st.sidebar.number_input("Number of Services", min_value=0.0, help="Enter the number of services provided.")
    number_of_medicare_beneficiaries = st.sidebar.number_input("Number of Medicare Beneficiaries", min_value=0, help="Enter the number of Medicare beneficiaries.")
    number_of_distinct_medicare_beneficiary_per_day_services = st.sidebar.number_input("Number of Distinct Medicare Beneficiary/Per Day Services", min_value=0, help="Enter the number of distinct Medicare beneficiary per day services.")
    average_medicare_allowed_amount = st.sidebar.number_input("Average Medicare Allowed Amount", min_value=0.0, help="Enter the average Medicare allowed amount.")
    average_submitted_charge_amount = st.sidebar.number_input("Average Submitted Charge Amount", min_value=0.0, help="Enter the average submitted charge amount.")
    average_medicare_payment_amount = st.sidebar.number_input("Average Medicare Payment Amount", min_value=0.0, help="Enter the average Medicare payment amount.")
    average_medicare_standardized_amount = st.sidebar.number_input("Average Medicare Standardized Amount", min_value=0.0, help="Enter the average Medicare standardized amount.")

    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    X_train, X_test = train_test_split(df_encoded, test_size=0.2, random_state=42)

    iforest = IsolationForest(contamination=0.05)
    iforest.fit(X_train)

    if st.button("Submit"):
        user_input_df = pd.DataFrame({
            'Number of Services': [number_of_services],
            'Number of Medicare Beneficiaries': [number_of_medicare_beneficiaries],
            'Number of Distinct Medicare Beneficiary/Per Day Services': [number_of_distinct_medicare_beneficiary_per_day_services],
            'Average Medicare Allowed Amount': [average_medicare_allowed_amount],
            'Average Submitted Charge Amount': [average_submitted_charge_amount],
            'Average Medicare Payment Amount': [average_medicare_payment_amount],
            'Average Medicare Standardized Amount': [average_medicare_standardized_amount],
        })

        for col in categorical_columns:
            user_input_df[col] = 0
        
        user_input_encoded = pd.get_dummies(user_input_df, columns=categorical_columns)
        user_input_encoded = user_input_encoded.reindex(columns=X_train.columns, fill_value=0)

        user_anomaly = iforest.predict(user_input_encoded)
        user_anomaly[user_anomaly == 1] = 0
        user_anomaly[user_anomaly == -1] = 1

        if user_anomaly == 0:
            st.success("Prediction: No Suspected Fraud")
        else:
            st.error("Prediction: Suspected Fraud. Please investigate.")

if __name__ == "__main__":
    main()
