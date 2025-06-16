import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# Page config
st.set_page_config(page_title="Health Assistant", layout="centered", page_icon="🧑‍⚕️")

# Load model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "saved_models", "parkinsons_model.sav")
parkinsons_model = pickle.load(open(model_path, 'rb'))

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "Parkinson's Prediction System",
        ['Parkinsons Prediction'],
        menu_icon='hospital-fill',
        icons=['activity'],
        default_index=0)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("🧠 Parkinson's Disease Prediction")

    st.markdown("### Enter the patient's voice measurement features:")
    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            fo = st.text_input('MDVP:Fo(Hz)')
            fhi = st.text_input('MDVP:Fhi(Hz)')
            flo = st.text_input('MDVP:Flo(Hz)')
            Jitter_percent = st.text_input('MDVP:Jitter(%)')
            Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
            RAP = st.text_input('MDVP:RAP')
            PPQ = st.text_input('MDVP:PPQ')
            DDP = st.text_input('Jitter:DDP')
            Shimmer = st.text_input('MDVP:Shimmer')
            Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
            APQ3 = st.text_input('Shimmer:APQ3')

        with col2:
            APQ5 = st.text_input('Shimmer:APQ5')
            APQ = st.text_input('MDVP:APQ')
            DDA = st.text_input('Shimmer:DDA')
            NHR = st.text_input('NHR')
            HNR = st.text_input('HNR')
            RPDE = st.text_input('RPDE')
            DFA = st.text_input('DFA')
            spread1 = st.text_input('spread1')
            spread2 = st.text_input('spread2')
            D2 = st.text_input('D2')
            PPE = st.text_input('PPE')


        submit_button = st.form_submit_button(label="🧪 Get Parkinson's Test Result")

    if submit_button:
        try:
            input_data = [
                fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                RPDE, DFA, spread1, spread2, D2, PPE
            ]

            prediction = parkinsons_model.predict([input_data])

            if prediction == 1:
                st.error("⚠️ The person **has Parkinson's disease**.")
            else:
                st.success("✅ The person **does not have Parkinson's disease**.")

        except Exception as e:
            st.warning(f"An error occurred while making prediction: {e}")