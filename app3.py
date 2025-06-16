import streamlit as st
import numpy as np
import pickle

# Page configuration
st.set_page_config(page_title="Parkinson's Disease Predictor", layout="centered")

st.title("🧠 Parkinson's Disease Predictor")
st.markdown("Enter the required values and upload a trained model to predict whether the person has Parkinson's disease.")

# Collect user input for the features
features = [
    "fo", "fhi", "flo", "Jitter_percent", "Jitter_Abs", "RAP", "PPQ", "DDP",
    "Shimmer", "Shimmer_dB", "APQ3", "APQ5", "APQ", "DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

user_input = []
with st.form("prediction_form"):
    st.subheader("Input Parameters")
    for feature in features:
        val = st.number_input(f"{feature}", value=0.0, format="%.6f")
        user_input.append(val)

    model_file = st.file_uploader("Upload your trained model (.pkl file)", type=["pkl"])

    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    if model_file is None:
        st.error("Please upload a trained model file.")
    else:
        try:
            # Load model
            model = pickle.load(model_file)

            # Prepare data and predict
            data = np.array(user_input).reshape(1, -1)
            prediction = model.predict(data)[0]

            # Display result
            if prediction == 1:
                st.error("❗ The model predicts: Parkinson's disease detected.")
            else:
                st.success("✅ The model predicts: No Parkinson's disease.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
