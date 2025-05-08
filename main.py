# main.py

import streamlit as st
import pandas as pd
import joblib
import warnings

def run():
    st.set_page_config(
        page_title="Heart Failure Risk Predictor",
        page_icon="❤️",
        layout="centered",
    )
    warnings.simplefilter("ignore", category=FutureWarning)

    @st.cache_data
    def load_pipeline(path):
        return joblib.load(path)

    # Carica tutta la pipeline (preprocessing + modello)
    pipeline = load_pipeline("model/heart_failure_pipeline.pkl")

    st.title("Explainable Heart Failure Risk Prediction")
    st.sidebar.header("Patient Parameters")

    # --- Input form ---
    age             = st.sidebar.slider("Age", 1, 120, 50)
    sex             = st.sidebar.selectbox("Sex", ["Male", "Female"])
    chest_pain      = st.sidebar.selectbox(
                         "Chest Pain Type",
                         ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
                      )
    resting_bp      = st.sidebar.slider("Resting BP (mm Hg)", 50, 250, 120)
    cholesterol     = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fasting_bs      = st.sidebar.selectbox("Fasting BS >120 mg/dl", ["Yes", "No"])
    resting_ecg     = st.sidebar.selectbox(
                         "Resting ECG",
                         ["Normal", "ST-T Abnormality", "LVH by Estes"]
                      )
    max_hr          = st.sidebar.slider("Max HR", 60, 220, 150)
    exercise_angina = st.sidebar.selectbox("Exercise Angina", ["Yes", "No"])
    oldpeak         = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    st_slope        = st.sidebar.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

    if st.sidebar.button("Predict"):
        # 1) prepara il DataFrame grezzo
        df_input = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": 1 if fasting_bs=="Yes" else 0,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": 1 if exercise_angina=="Yes" else 0,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope
        }])

        # 2) lasciamo fare tutto alla pipeline
        pred  = pipeline.predict(df_input)[0]
        proba = pipeline.predict_proba(df_input)[0]

        # 3) mostra risultato
        st.subheader("Prediction Result")
        if pred == 1:
            st.error("⚠️ Patient is **at risk** of heart failure")
        else:
            st.success("✅ Patient is **not at risk**")

        st.markdown("**Predicted Probabilities:**")
        st.write(f"- Not at risk: {proba[0]*100:.2f}%")
        st.write(f"- At risk: {proba[1]*100:.2f}%")

if __name__ == "__main__":
    run()
