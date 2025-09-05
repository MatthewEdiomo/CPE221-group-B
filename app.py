import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configure page
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the trained model and feature columns"""
    try:
        model = joblib.load('model.joblib')
        model_columns = joblib.load('model_columns.joblib')
        return model, model_columns
    except FileNotFoundError:
        st.error("Model files not found. Please ensure model files are available.")
        return None, None

# Load model
model, model_columns = load_model()

# Simple sidebar
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This tool predicts laptop prices based on specifications.
    
    Simply select your desired features and get an instant price estimate.
    
    The prediction is based on machine learning analysis of laptop market data.
    """)

# Main header
st.title("💻 Laptop Price Predictor")
st.markdown("Enter laptop specifications below:")

if model is None or model_columns is None:
    st.stop()

# Simple form layout
st.markdown("### Specifications")

# Row 1
col1, col2 = st.columns(2)
with col1:
    company = st.selectbox('Brand', ['Acer', 'Apple', 'Asus', 'Dell', 'HP', 'Lenovo', 'MSI', 'Other'])
with col2:
    typename = st.selectbox('Type', ['Notebook', 'Gaming', 'Ultrabook', '2 in 1 Convertible', 'Workstation', 'Netbook'])

# Row 2
col3, col4 = st.columns(2)
with col3:
    ram = st.selectbox('RAM', [2, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64])
with col4:
    storage_size_gb = st.selectbox('Storage Size (GB)', [64, 128, 256, 512, 1024, 2048])

# Row 3
col5, col6 = st.columns(2)
with col5:
    inches = st.selectbox('Screen Size', [10.1, 11.6, 12.5, 13.3, 14.0, 15.6, 17.3, 18.4])
with col6:
    weight = st.selectbox('Weight (kg)', [0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 4.5])

# Row 4
col7, col8 = st.columns(2)
with col7:
    storage_type = st.selectbox('Storage Type', ['SSD', 'HDD', 'Flash', 'Hybrid'])
with col8:
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# Row 5
ips_panel = st.selectbox('IPS Panel', ['No', 'Yes'])

# Simple predict button
st.markdown("---")
if st.button('Predict Price', type="primary", use_container_width=True):
    try:
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
        
        # One-hot encode the categorical features
        company_encoded = pd.get_dummies(pd.Series([company]), prefix='Company', drop_first=True)
        typename_encoded = pd.get_dummies(pd.Series([typename]), prefix='TypeName', drop_first=True)
        storage_encoded = pd.get_dummies(pd.Series([storage_type]), prefix='Storage_Type', drop_first=True)
        
        # Align with model columns
        final_df = pd.DataFrame(0, index=input_df.index, columns=model_columns)
        final_df.update(input_df)
        
        # Update one-hot encoded columns
        for col in company_encoded.columns:
            if col in final_df.columns:
                final_df[col] = company_encoded[col].values[0]
        
        for col in typename_encoded.columns:
            if col in final_df.columns:
                final_df[col] = typename_encoded[col].values[0]
        
        for col in storage_encoded.columns:
            if col in final_df.columns:
                final_df[col] = storage_encoded[col].values[0]
        
        # Make prediction
        prediction = model.predict(final_df)
        
        # Display result
        st.success(f"**Predicted Price: €{prediction[0]:,.2f}**")
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")