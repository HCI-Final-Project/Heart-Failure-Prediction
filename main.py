import streamlit as st
import pandas as pd
import joblib
import warnings
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def run():
    st.set_page_config(
        page_title="Heart Failure Risk Predictor",
        page_icon="❤️",
        layout="centered",
    )
    warnings.simplefilter("ignore", category=FutureWarning)

    @st.cache_data
    def load_model(path):
        return joblib.load(path)

    # Carica solo il modello
    model = load_model("model/heart_disease_model.pkl")

    st.title("Explainable Heart Failure Risk Prediction")
    st.sidebar.header("Patient Parameters")

    # --- Input form ---
    age = st.sidebar.slider("Age", 18, 120, 50)  
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.sidebar.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
    )
    resting_bp = st.sidebar.slider("Resting BP (mm Hg)", 80, 200, 120)  
    cholesterol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 400, 200)  
    fasting_bs = st.sidebar.selectbox("Fasting BS >120 mg/dl", ["Yes", "No"])
    resting_ecg = st.sidebar.selectbox(
        "Resting ECG",
        ["Normal", "ST-T Abnormality", "LVH by Estes"]
    )
    max_hr = st.sidebar.slider("Max HR", 60, 200, 150)  
    exercise_angina = st.sidebar.selectbox("Exercise Angina", ["Yes", "No"])
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 4.0, 1.0, step=0.1) 
    st_slope = st.sidebar.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

    if st.sidebar.button("Predict"):
        # 1) prepara il DataFrame grezzo
        df_input = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": 1 if fasting_bs == "Yes" else 0,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope
        }])

        # 2) Preprocessing manuale
        # Standardizzazione delle colonne numeriche
        numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
        scaler = StandardScaler()
        df_input[numeric_cols] = scaler.fit_transform(df_input[numeric_cols])
        
        # One-Hot Encoding per le colonne categoriche
        string_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        encoded_df = pd.DataFrame()
        
        for col in string_columns:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(df_input[[col]])
            cols = encoder.get_feature_names_out([col])
            encoded_part = pd.DataFrame(encoded, columns=cols, index=df_input.index)
            encoded_df = pd.concat([encoded_df, encoded_part], axis=1)
        
        # Unione delle colonne
        df_input = df_input.drop(columns=string_columns)
        df_input = pd.concat([df_input, encoded_df], axis=1)
        
        # Assicurati che tutte le colonne attese siano presenti
        expected_columns = [
            'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
            'Sex_F', 'Sex_M', 'ChestPainType_ASY', 'ChestPainType_ATA', 
            'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_LVH', 
            'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_N', 
            'ExerciseAngina_Y', 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up'
        ]
        
        # Aggiungi eventuali colonne mancanti con valore 0
        for col in expected_columns:
            if col not in df_input.columns:
                df_input[col] = 0
        
        # Riordina le colonne come nel training
        df_input = df_input[expected_columns]

        # 3) Predizione
        pred = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0]

        # 4) mostra risultato
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