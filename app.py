import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and feature columns
model = joblib.load(r'model.joblib')
model_columns = joblib.load(r'model_columns.joblib')

# Set up the Streamlit app layout
st.title('Laptop Price Predictor')
st.markdown('Enter the laptop specifications to get a price prediction.')

# Create input widgets for user features
company = st.selectbox('Company', ['Acer', 'Apple', 'Asus', 'Dell', 'HP', 'Lenovo', 'MSI', 'Other'])
typename = st.selectbox('Type', ['Notebook', 'Gaming', 'Ultrabook', '2 in 1 Convertible', 'Workstation', 'Netbook'])
ram = st.slider('RAM (in GB)', 2, 64, 8, step=2)
inches = st.slider('Screen Size (in Inches)', 10.1, 18.4, 15.6)
weight = st.slider('Weight (in kg)', 0.69, 4.7, 2.0)
storage_type = st.selectbox('Storage Type', ['SSD', 'HDD', 'Flash', 'Hybrid'])
storage_size_gb = st.slider('Storage Size (in GB)', 64, 2048, 256, step=64)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips_panel = st.selectbox('IPS Panel', ['No', 'Yes'])

# Convert user inputs into a DataFrame
input_data = {
    'Inches': [inches],
    'Ram': [ram],
    'Weight': [weight],
    'Touchscreen': [1 if touchscreen == 'Yes' else 0],
    'IPS_Panel': [1 if ips_panel == 'Yes' else 0],
    'Storage_Size_GB': [storage_size_gb],
}

# Create a DataFrame from the inputs
input_df = pd.DataFrame(input_data)

# One-hot encode the categorical features from user input
company_encoded = pd.get_dummies(pd.Series([company]), prefix='Company', drop_first=True)
typename_encoded = pd.get_dummies(pd.Series([typename]), prefix='TypeName', drop_first=True)
storage_encoded = pd.get_dummies(pd.Series([storage_type]), prefix='Storage_Type', drop_first=True)

# Align the input DataFrame with the model's training columns
# This is a crucial step to ensure the one-hot encoded columns are in the correct order
final_df = pd.DataFrame(0, index=input_df.index, columns=model_columns)
final_df.update(input_df)

# Update the one-hot encoded columns
for col in company_encoded.columns:
    if col in final_df.columns:
        final_df[col] = company_encoded[col].values[0]
for col in typename_encoded.columns:
    if col in final_df.columns:
        final_df[col] = typename_encoded[col].values[0]
for col in storage_encoded.columns:
    if col in final_df.columns:
        final_df[col] = storage_encoded[col].values[0]

# Prediction button
if st.button('Predict Price'):
    # Make prediction
    try:
        prediction = model.predict(final_df)
        st.success(f'The predicted price is: **€{prediction[0]:.2f}**')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")