import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import platform

# Check for required packages
try:
    import xgboost
    from joblib import load
    XGBOOST_AVAILABLE = True
except ImportError as e:
    XGBOOST_AVAILABLE = False
    st.error(f"""
    Required packages not found. Please install them by running:
    `pip install xgboost scikit-learn joblib`
    
    Error details: {str(e)}
    """)
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Parkinson's Disease Prediction",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .stNumberInput input {
        font-size: 14px;
    }
    .st-b7 {
        color: white;
    }
    .st-c0 {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        model = load('saved_models/parkinsons_model.joblib')
        scaler = load('saved_models/parkinsons_scaler.joblib')
        return model, scaler
    except Exception as e:
        st.error(f"""
        Error loading model files. Please ensure:
        1. 'parkinsons_model.joblib' exists
        2. 'parkinsons_scaler.joblib' exists
        3. Files are in the same directory as this script
        
        Error details: {str(e)}
        """)
        st.stop()

# Load models
model, scaler = load_models()

# App title and description
st.title("🧠 Parkinson's Disease Prediction System")
st.markdown("""
This app predicts the likelihood of Parkinson's disease based on voice measurements.
Please enter all the required features below and click 'Predict'.
""")

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fundamental Frequency Features")
        mdvp_fo = st.number_input("MDVP:Fo(Hz)", min_value=50.0, max_value=300.0, value=120.0, step=0.1)
        mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", min_value=50.0, max_value=600.0, value=150.0, step=0.1)
        mdvp_flo = st.number_input("MDVP:Flo(Hz)", min_value=50.0, max_value=300.0, value=80.0, step=0.1)
        
        st.subheader("Jitter Features")
        mdvp_jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, max_value=0.1, value=0.005, step=0.00001, format="%.5f")
        mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, max_value=0.001, value=0.00005, step=0.000001, format="%.6f")
        mdvp_rap = st.number_input("MDVP:RAP", min_value=0.0, max_value=0.1, value=0.003, step=0.00001, format="%.5f")
        mdvp_ppq = st.number_input("MDVP:PPQ", min_value=0.0, max_value=0.1, value=0.004, step=0.00001, format="%.5f")
        jitter_ddp = st.number_input("Jitter:DDP", min_value=0.0, max_value=0.2, value=0.01, step=0.00001, format="%.5f")
        
    with col2:
        st.subheader("Shimmer Features")
        mdvp_shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=0.2, value=0.03, step=0.00001, format="%.5f")
        mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, max_value=1.5, value=0.3, step=0.001, format="%.3f")
        shimmer_apq3 = st.number_input("Shimmer:APQ3", min_value=0.0, max_value=0.1, value=0.02, step=0.00001, format="%.5f")
        shimmer_apq5 = st.number_input("Shimmer:APQ5", min_value=0.0, max_value=0.1, value=0.03, step=0.00001, format="%.5f")
        mdvp_apq = st.number_input("MDVP:APQ", min_value=0.0, max_value=0.1, value=0.03, step=0.00001, format="%.5f")
        shimmer_dda = st.number_input("Shimmer:DDA", min_value=0.0, max_value=0.2, value=0.06, step=0.00001, format="%.5f")
        
        st.subheader("Other Features")
        nhr = st.number_input("NHR", min_value=0.0, max_value=0.5, value=0.02, step=0.00001, format="%.5f")
        hnr = st.number_input("HNR", min_value=0.0, max_value=40.0, value=20.0, step=0.001, format="%.3f")
    
    # Additional features
    st.subheader("Nonlinear Features")
    col3, col4 = st.columns(2)
    with col3:
        rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0, value=0.4, step=0.000001, format="%.6f")
        dfa = st.number_input("DFA", min_value=0.0, max_value=1.0, value=0.7, step=0.000001, format="%.6f")
    with col4:
        spread1 = st.number_input("spread1", min_value=-10.0, max_value=0.0, value=-5.0, step=0.000001, format="%.6f")
        spread2 = st.number_input("spread2", min_value=0.0, max_value=0.5, value=0.2, step=0.000001, format="%.6f")
    
    col5, col6 = st.columns(2)
    with col5:
        d2 = st.number_input("D2", min_value=1.0, max_value=5.0, value=2.0, step=0.000001, format="%.6f")
    with col6:
        ppe = st.number_input("PPE", min_value=0.0, max_value=0.5, value=0.2, step=0.000001, format="%.6f")
    
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        'MDVP:Fo(Hz)': mdvp_fo,
        'MDVP:Fhi(Hz)': mdvp_fhi,
        'MDVP:Flo(Hz)': mdvp_flo,
        'MDVP:Jitter(%)': mdvp_jitter_percent,
        'MDVP:Jitter(Abs)': mdvp_jitter_abs,
        'MDVP:RAP': mdvp_rap,
        'MDVP:PPQ': mdvp_ppq,
        'Jitter:DDP': jitter_ddp,
        'MDVP:Shimmer': mdvp_shimmer,
        'MDVP:Shimmer(dB)': mdvp_shimmer_db,
        'Shimmer:APQ3': shimmer_apq3,
        'Shimmer:APQ5': shimmer_apq5,
        'MDVP:APQ': mdvp_apq,
        'Shimmer:DDA': shimmer_dda,
        'NHR': nhr,
        'HNR': hnr,
        'RPDE': rpde,
        'DFA': dfa,
        'spread1': spread1,
        'spread2': spread2,
        'D2': d2,
        'PPE': ppe
    }
    
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    
    st.subheader("Results")
    if prediction == 1:
        st.error("### 🚨 Prediction: Parkinson's Disease Detected")
    else:
        st.success("### ✅ Prediction: No Parkinson's Disease Detected")
    
    st.metric("Probability of Parkinson's", f"{proba[1]*100:.2f}%")
    st.metric("Probability of Being Healthy", f"{proba[0]*100:.2f}%")
    
    # Feature importance visualization
    try:
        st.subheader("Feature Importance")
        importance = model.feature_importances_
        features = input_df.columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))
    except Exception as e:
        st.warning("Could not display feature importance. Some models don't support this feature.")

# Sidebar information
st.sidebar.header("About")
st.sidebar.info("""
This prediction system uses machine learning to analyze voice measurements.

**Model Information:**
- Algorithm: XGBoost
- Accuracy: ~95% (on test data)
- Input Features: 22 voice parameters

**How to use:**
1. Fill in all measurement values
2. Click 'Predict'
3. View results and probabilities
""")

st.sidebar.markdown("""
**Note:** This is a predictive tool, not a medical diagnosis. 
Always consult a healthcare professional for medical advice.
""")