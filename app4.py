import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Set page title and icon
st.set_page_config(page_title="Parkinson's Disease Prediction", page_icon="🏥")

# Load the trained model and scaler
@st.cache_resource
def load_model():
    with open('saved_models/parkinsons_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('saved_models/parkinsons_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

model, scaler = load_model()

# App title and description
st.title("""🧠 Parkinson's Disease Prediction System""")
st.markdown("""
This app predicts the likelihood of Parkinson's disease based on voice measurements.
Please enter all the required features below and click 'Predict'.
""")

import streamlit as st

# Create input form
with st.form("prediction_form"):
    st.header("Patient Information")
    
    # Divide inputs into columns for better layout
    col1, col2, col3 = st.columns(3)  # Adjusted to 3 columns for the 3 groups
    
    with col1:
        st.subheader("Fundamental Frequency Features")
        mdvp_fo = st.number_input("MDVP:Fo(Hz)", min_value=50.0, max_value=300.0, value=120.0)
        mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", min_value=50.0, max_value=600.0, value=150.0)
        mdvp_flo = st.number_input("MDVP:Flo(Hz)", min_value=50.0, max_value=300.0, value=80.0)
        
        st.subheader("Additional Nonlinear Features")
        spread1 = st.number_input("spread1", min_value=-10.0, max_value=0.0, value=-5.0, format="%.6f")
        spread2 = st.number_input("spread2", min_value=0.0, max_value=0.5, value=0.2, format="%.6f")
        d2 = st.number_input("D2", min_value=1.0, max_value=5.0, value=2.0, format="%.6f")
        ppe = st.number_input("PPE", min_value=0.0, max_value=0.5, value=0.2, format="%.6f")
    
    with col2:
        st.subheader("Shimmer Features")
        mdvp_shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=0.2, value=0.03, format="%.5f")
        mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, max_value=1.5, value=0.3, format="%.3f")
        shimmer_apq3 = st.number_input("Shimmer:APQ3", min_value=0.0, max_value=0.1, value=0.02, format="%.5f")
        shimmer_apq5 = st.number_input("Shimmer:APQ5", min_value=0.0, max_value=0.1, value=0.03, format="%.5f")
        mdvp_apq = st.number_input("MDVP:APQ", min_value=0.0, max_value=0.1, value=0.03, format="%.5f")
        shimmer_dda = st.number_input("Shimmer:DDA", min_value=0.0, max_value=0.2, value=0.06, format="%.5f")

        st.subheader("Other Features")
        nhr = st.number_input("NHR", min_value=0.0, max_value=0.5, value=0.02, format="%.5f")
        hnr = st.number_input("HNR", min_value=0.0, max_value=40.0, value=20.0, format="%.3f")
    
    with col3:
        st.subheader("Jitter Features")
        mdvp_jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, max_value=0.1, value=0.005, format="%.5f")
        mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, max_value=0.001, value=0.00005, format="%.6f")
        mdvp_rap = st.number_input("MDVP:RAP", min_value=0.0, max_value=0.1, value=0.003, format="%.5f")
        mdvp_ppq = st.number_input("MDVP:PPQ", min_value=0.0, max_value=0.1, value=0.004, format="%.5f")
        jitter_ddp = st.number_input("Jitter:DDP", min_value=0.0, max_value=0.2, value=0.01, format="%.5f")
        
        st.subheader("Nonlinear Features")
        rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0, value=0.4, format="%.6f")
        dfa = st.number_input("DFA", min_value=0.0, max_value=1.0, value=0.7, format="%.6f")
    
    # Prediction button
    submitted = st.form_submit_button("Predict")   
# When form is submitted
if submitted:
    # Create dictionary of input features
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
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Display results
    st.subheader("Results")
    
    if prediction == 1:
        st.error("🚨 Prediction: The person has Parkinson's disease")
    else:
        st.success("✅ Prediction: The person does not have Parkinson's disease")
    
    st.write(f"Probability of having Parkinson's: {prediction_proba[1]*100:.2f}%")
    st.write(f"Probability of being healthy: {prediction_proba[0]*100:.2f}%")
    
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