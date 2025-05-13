import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import warnings
import shap                                 # ── SHAP: explainer e plots
import matplotlib.pyplot as plt             # ── SHAP: fallback matplotlib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ── MAPPING FEATURE → ETICHETTA UMAN-FRIENDLY
FEATURE_LABELS = {
    "Age":               "Age",
    "RestingBP":         "Resting blood pressure",
    "Cholesterol":       "Cholesterol",
    "FastingBS":         "Fasting blood sugar",
    "MaxHR":             "Maximum heart rate",
    "Oldpeak":           "ST depression",
    "Sex_F":             "Sex = Female",
    "Sex_M":             "Sex = Male",
    "ChestPainType_ASY": "Asymptomatic pain",
    "ChestPainType_ATA": "Atypical pain",
    "ChestPainType_NAP": "Non-anginal pain",
    "ChestPainType_TA":  "Typical pain",
    "RestingECG_LVH":    "ECG LVH",
    "RestingECG_Normal": "Normal ECG",
    "RestingECG_ST":     "ECG ST-T",
    "ExerciseAngina_N":  "No angina",
    "ExerciseAngina_Y":  "Exercise-induced angina",
    "ST_Slope_Down":     "ST Slope: Down",
    "ST_Slope_Flat":     "ST Slope: Flat",
    "ST_Slope_Up":       "ST Slope: Up",
}

def st_shap(plot, height=None, width=None):
    """Integra un force-plot SHAP (matplotlib=False) in Streamlit."""
    shap_js   = shap.getjs()
    plot_html = plot.html()
    html      = f"<head>{shap_js}</head><body>{plot_html}</body>"
    components.html(html, height=height, width=width)

def run():
    st.set_page_config(
        page_title="Explainable Heart Failure Risk Predictor",
        page_icon="❤️",
        layout="wide",   # ── LAYOUT WIDE per tutto lo spazio disponibile
    )
    warnings.simplefilter("ignore", category=FutureWarning)

    @st.cache_data
    def load_model(path):
        return joblib.load(path)

    @st.cache_data
    def load_background(csv_path, n=100):
        df_bg = pd.read_csv(csv_path).head(n)
        # preprocessing identico a df_input
        num_cols = ['Age','RestingBP','Cholesterol','MaxHR']
        df_bg[num_cols] = StandardScaler().fit_transform(df_bg[num_cols])
        cats = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
        enc_df = pd.DataFrame(index=df_bg.index)
        for col in cats:
            enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            mat = enc.fit_transform(df_bg[[col]])
            names = enc.get_feature_names_out([col])
            enc_df = pd.concat([enc_df, pd.DataFrame(mat, columns=names, index=df_bg.index)], axis=1)
        proc = pd.concat([df_bg.drop(columns=cats), enc_df], axis=1)
        # allineo colonne
        expected = [
            'Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak',
            'Sex_F','Sex_M','ChestPainType_ASY','ChestPainType_ATA','ChestPainType_NAP','ChestPainType_TA',
            'RestingECG_LVH','RestingECG_Normal','RestingECG_ST',
            'ExerciseAngina_N','ExerciseAngina_Y','ST_Slope_Down','ST_Slope_Flat','ST_Slope_Up'
        ]
        for c in expected:
            if c not in proc: proc[c] = 0
        return proc[expected].values

    # CARICAMENTO
    model = load_model("model/heart_disease_model.pkl")
    background_data = load_background("model/heart.csv")

    st.title("Explainable Heart Failure Risk Prediction")
    st.sidebar.header("Patient Parameters")

    # --- Input form ---
    age = st.sidebar.slider("Age", 18, 120, 50)
    sex = st.sidebar.selectbox("Sex", ["Male","Female"])
    chest_pain = st.sidebar.selectbox("Chest Pain Type",
        ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"])
    resting_bp = st.sidebar.slider("Resting BP (mm Hg)", 80,200,120)
    cholesterol = st.sidebar.slider("Cholesterol (mg/dl)",100,400,200)
    fasting_bs = st.sidebar.selectbox("Fasting BS >120 mg/dl", ["Yes","No"])
    resting_ecg = st.sidebar.selectbox("Resting ECG",
        ["Normal","ST-T Abnormality","LVH by Estes"])
    max_hr = st.sidebar.slider("Max HR", 60,200,150)
    exercise_angina = st.sidebar.selectbox("Exercise Angina", ["Yes","No"])
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0,4.0,1.0, step=0.1)
    st_slope = st.sidebar.selectbox("ST Slope", ["Upsloping","Flat","Downsloping"])

    if st.sidebar.button("Predict"):
        # --- PREPROCESSING INPUT identico a prima ---
        df_input = pd.DataFrame([{
            "Age":age, "Sex":sex, "ChestPainType":chest_pain,
            "RestingBP":resting_bp, "Cholesterol":cholesterol,
            "FastingBS":1 if fasting_bs=="Yes" else 0,
            "RestingECG":resting_ecg, "MaxHR":max_hr,
            "ExerciseAngina":exercise_angina, "Oldpeak":oldpeak,
            "ST_Slope":st_slope
        }])
        num_cols = ['Age','RestingBP','Cholesterol','MaxHR']
        df_input[num_cols] = StandardScaler().fit_transform(df_input[num_cols])
        cats = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
        enc_df = pd.DataFrame()
        for col in cats:
            enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            mat = enc.fit_transform(df_input[[col]])
            names = enc.get_feature_names_out([col])
            enc_df = pd.concat([enc_df, pd.DataFrame(mat,columns=names,index=df_input.index)], axis=1)
        df_input = pd.concat([df_input.drop(columns=cats), enc_df], axis=1)
        expected = [
            'Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak',
            'Sex_F','Sex_M','ChestPainType_ASY','ChestPainType_ATA','ChestPainType_NAP','ChestPainType_TA',
            'RestingECG_LVH','RestingECG_Normal','RestingECG_ST',
            'ExerciseAngina_N','ExerciseAngina_Y','ST_Slope_Down','ST_Slope_Flat','ST_Slope_Up'
        ]
        for c in expected:
            if c not in df_input: df_input[c] = 0
        df_input = df_input[expected]

        # 3) PREDIZIONE
        pred  = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0]

        # 4) RISULTATO
        st.subheader("Prediction Result")
        if pred==1:
            st.error("⚠️ Patient is **at risk** of heart failure")
        else:
            st.success("✅ Patient is **not at risk**")
        c1,c2 = st.columns(2)
        c1.metric("Not at risk", f"{proba[0]*100:.2f} %")
        c2.metric("At risk",     f"{proba[1]*100:.2f} %")

        # ── SHAP EXPLANATION ──
        explainer   = shap.Explainer(model, background_data)
        sv          = explainer(df_input)
        # estraggo shap values e baseline e li normalizzo a percentuale
        vals  = sv.values[0, :, 1] * 100    # ora in punti percentuali
        base  = sv.base_values[0, 1] * 100
        data  = sv.data[0]
        names = sv.feature_names
        # top-K
        import numpy as np
        idx = np.argsort(np.abs(vals))[::-1][:7]
        vals_top  = vals[idx]
        data_top  = data[idx]
        names_top = [names[i] for i in idx]

        # etichette friendly

        pretty = []
        for feat in names_top:
            pretty.append(FEATURE_LABELS.get(feat, feat))
        from shap import Explanation
        single_exp = Explanation(
            values=vals_top,
            base_values=base,
            data=data_top,
            feature_names=pretty
        )

        # legenda colori
        st.markdown("""
        <div style="display:flex; justify-content:center; gap:2rem;">
          <span style="color:#E74C3C;">&#9632; incrise probability</span>
          <span style="color:#3498DB;">&#9632; decrise probability</span>
        </div>
        """, unsafe_allow_html=True)

        # force plot più largo
        # margine in alto per distanziare dal titolo/metriche
        st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)

        force_plot = shap.plots.force(
            explainer.expected_value[1] * 100,     # baseline in %
            single_exp.values,                     # già in %
            single_exp.data,
            feature_names=single_exp.feature_names,
            matplotlib=False
        )
        st_shap(force_plot, height=300, width=1000)
 
        # baseline & prediction chiaramente
        # esplicito in chiaro che si tratta di percentuali
        st.markdown(f""" **Baseline probability:** {base:.2f}%   **Predicted probability (at risk):** {proba[1]*100:.2f}% """)

if __name__ == "__main__":
    run()
