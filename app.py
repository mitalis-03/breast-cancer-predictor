import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Cancer Diagnostic Tool",
    page_icon="🩺",
    layout="wide"
)

# 2. Custom CSS for Professional Look
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #004aad;
        color: white;
    }
    .stButton>button:hover {
        background-color: #003073;
        color: white;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #d1d8e0;
    }
    .report-card {
        padding: 25px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Model & Scaler
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('breast_cancer_model.joblib')
        scaler = joblib.load('breast_cancer_scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model or Scaler file not found. Please ensure .joblib files are in the same directory.")
        return None, None

model, scaler = load_assets()

# 4. Header Section
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2864/2864230.png", width=100)
with col2:
    st.title("Breast Cancer Diagnostic Assistant")
    st.markdown("##### *Precision Medicine Powered by Random Forest Machine Learning*")

st.divider()

# 5. Sidebar Inputs
st.sidebar.markdown("### 📊 Biopsy Data Input")
st.sidebar.write("Input measurements from the laboratory report.")

def get_clean_user_inputs(scaler):
    feature_names = scaler.get_feature_names_out()
    user_inputs = {}
    
    # We create two columns in the sidebar for a compact look
    for feat in feature_names:
        user_inputs[feat] = st.sidebar.number_input(
            f"{feat.replace('_', ' ').capitalize()}", 
            step=0.1,
            format="%.2f"
        )
    return pd.DataFrame([user_inputs])

input_df = get_clean_user_inputs(scaler)

# 6. Main Content Area
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Data Summary")
    st.info("The values entered in the sidebar are summarized below for verification.")
    st.dataframe(input_df.T.rename(columns={0: 'Value'}), use_container_width=True)

with col_right:
    st.subheader("Diagnostic Engine")
    st.write("Click below to run the diagnosis through the trained model.")
    
    if st.button("Run Diagnostic Analysis", type="primary"):
        if model and scaler:
            # Prediction Logic
            scaled_data = scaler.transform(input_df)
            prediction = model.predict(scaled_data)[0]
            prob = model.predict_proba(scaled_data)[0]
            
            st.markdown("<div class='report-card'>", unsafe_allow_html=True)
            if prediction == 0:
                st.markdown(f"""
                    <div style="background-color: #ffebee; border-left: 10px solid #c62828; padding: 20px; border-radius: 10px;">
                        <h2 style="color: #c62828; margin:0;">Result: MALIGNANT</h2>
                        <p style="color: #444; margin-top:10px;">The model predicts a high probability of malignancy.</p>
                        <h3 style="color: #c62828; margin:0;">Confidence: {prob[0]:.2%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="background-color: #e8f5e9; border-left: 10px solid #2e7d32; padding: 20px; border-radius: 10px;">
                        <h2 style="color: #2e7d32; margin:0;">Result: BENIGN</h2>
                        <p style="color: #444; margin-top:10px;">The model predicts the tumor is likely non-cancerous.</p>
                        <h3 style="color: #2e7d32; margin:0;">Confidence: {prob[1]:.2%}</h3>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
st.divider()
st.caption("⚠️ **Disclaimer:** This tool is for educational purposes only. Always consult with a qualified medical professional for clinical diagnosis.")
