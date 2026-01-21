import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Breast Cancer Diagnostic System", layout="wide")

# 2. Load Model & Scaler (Ensure these files are in the same folder)
@st.cache_resource # Caches the model so it doesn't reload on every slider move
def load_assets():
    model = joblib.load('breast_cancer_model.joblib')
    scaler = joblib.load('breast_cancer_scaler.joblib')
    return model, scaler

model, scaler = load_assets()

# 3. Sidebar UI - Dynamic Input Sliders
st.sidebar.header("🔬 Biopsy Measurements")
st.sidebar.write("Adjust the values based on the lab report.")

def get_clean_user_inputs(scaler):
    # This automatically gets the names of features the model expects
    feature_names = scaler.get_feature_names_out()
    user_inputs = {}

    for feat in feature_names:
        # We create a slider for each feature. 
        # Note: 0.0 to 1.0 is a safe range for many features, 
        # but you can adjust these based on your df_reduced.describe()
        user_inputs[feat] = st.sidebar.slider(
            f"{feat.capitalize()}", 
            min_value=0.0, 
            max_value=100.0, # Adjust max based on your data spread
            value=20.0       # Default starting point
        )
    
    return pd.DataFrame([user_inputs])

# Get inputs from user
input_df = get_clean_user_inputs(scaler)

# 4. Main Page UI
st.title("🩺 Breast Cancer Diagnostic Assistant")
st.markdown("""
This tool uses a **Random Forest Classifier** to predict whether a tumor is **Malignant** or **Benign**.
Fill in the metrics in the sidebar and click the button below for the diagnosis.
""")

# Show a preview of the entered data
with st.expander("View Input Data Summary"):
    st.table(input_df)

# 5. Prediction Logic
if st.button("Run Diagnostic Analysis", type="primary"):
    # Apply the same scaling used during training
    scaled_data = scaler.transform(input_df)
    
    # Get Prediction and Probabilities
    prediction = model.predict(scaled_data)[0]
    probabilities = model.predict_proba(scaled_data)[0]
    
    st.divider()
    
    if prediction == 0:
        st.error("## 🚨 Result: MALIGNANT")
        st.write(f"The model is **{probabilities[0]:.2%}** confident that this sample indicates a cancerous tumor.")
    else:
        st.success("## ✅ Result: BENIGN")
        st.write(f"The model is **{probabilities[1]:.2%}** confident that this sample indicates a healthy tumor.")
    
    st.info("**Disclaimer:** This is a machine learning tool and should not replace professional medical advice.")