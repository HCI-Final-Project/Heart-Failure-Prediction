import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import warnings
import shap                                 
import matplotlib.pyplot as plt             
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import lime
import lime.lime_tabular
import plotly.graph_objects as go
from shap import Explanation
import plotly.express as px
import streamlit_toggle as tog

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
    if 'predicted' not in st.session_state:
        st.session_state.predicted = False
    
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

    # --- Input form with tooltips ---
    # --- Input form with clinically precise tooltips ---
    # nella run(), subito dopo st.sidebar.header(...)

    # 📌 dentro run(), al posto di st.sidebar.radio…
    with st.sidebar:
        advanced = tog.st_toggle_switch(
            label="Active Advanced mode",      # etichetta dello switch
            key="mode_switch",          # chiave unica
            default_value=False,        # False → Base, True → Advanced
            label_after=False,          # False: label prima dello switch
            inactive_color="#D3D3D3",   # colore quando off
            active_color="#11567F",     # colore quando on
            track_color="#29B5E8"       # colore della traccia
        )
    mode = "Advanced" if advanced else "Base"



    # --- Definizione degli input secondo la modalità ---
    if mode == "Advanced":
        # Input “classici” con slider / selectbox
        age = st.sidebar.slider("Age", 18, 120, 50, help="Patient age in completed years")
        resting_bp = st.sidebar.slider("Resting BP (mm Hg)", 80, 200, 120, help="Resting systolic BP")
        cholesterol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 400, 200, help="Serum total cholesterol")
        max_hr = st.sidebar.slider("Max HR", 60, 200, 150, help="Peak exercise heart rate")
        oldpeak = st.sidebar.slider(
            "ST Depression (Oldpeak)", 0.0, 4.0, 1.0, step=0.1,
            help="Magnitude of ST-segment depression"
        )
    else:
        # Modalità BASE: scegli un intervallo, poi prendi il valore medio
        def range_midpoint(label, options, help_text):
            """Helper: selectbox di ranges e ritorna punto medio"""
            sel = st.sidebar.selectbox(label, options, help=help_text)
            low, high = map(float, sel.split("-"))
            return (low + high) / 2

        age = range_midpoint(
            "Age range",
            ["18-40", "40-60", "60-80", "80-120"],
            "Select age interval; midpoint used as input"
        )
        resting_bp = range_midpoint(
            "Resting BP range (mm Hg)",
            ["80-100", "100-120", "120-140", "140-200"],
            "Select resting systolic BP interval"
        )
        cholesterol = range_midpoint(
            "Cholesterol range (mg/dl)",
            ["100-200", "200-300", "300-400"],
            "Select cholesterol interval"
        )
        max_hr = range_midpoint(
            "Max HR range",
            ["60-100", "100-140", "140-200"],
            "Select max heart rate interval"
        )
        oldpeak = range_midpoint(
            "ST Depression range (Oldpeak)",
            ["0.0-1.0", "1.0-2.0", "2.0-3.0", "3.0-4.0"],
            "Select ST depression interval"
        )

    # input categorici invariati
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"], help="Patient biological sex")
    chest_pain = st.sidebar.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
        help="Chest pain classification"
    )
    fasting_bs = st.sidebar.selectbox("Fasting BS >120 mg/dl", ["Yes", "No"], help="Elevated fasting glucose")
    resting_ecg = st.sidebar.selectbox(
        "Resting ECG",
        ["Normal", "ST-T Abnormality", "LVH by Estes"],
        help="Resting ECG interpretation"
    )
    exercise_angina = st.sidebar.selectbox("Exercise Angina", ["Yes", "No"], help="Exercise-induced angina")
    st_slope = st.sidebar.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"], help="Slope of ST segment")


    if st.sidebar.button("Predict"):
        st.session_state.predicted = True

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


        idx = np.argsort(np.abs(vals))[::-1]
        vals_top  = vals[idx]
        data_top  = data[idx]
        names_top = [names[i] for i in idx]

        # etichette friendly

        pretty = []
        for feat in names_top:
            pretty.append(FEATURE_LABELS.get(feat, feat))
        single_exp = Explanation(
            values=vals_top,
            base_values=base,
            data=data_top,
            feature_names=pretty
        )


        # --- prepara l’explainer LIME (come già fatto) ---
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=background_data,
            feature_names=sv.feature_names,
            class_names=['Not at risk', 'At risk'],
            mode='classification'
        )

        # --- spiega l’istanza ---
        i = 0
        exp = lime_explainer.explain_instance(
            df_input.iloc[i],            # DataFrameRow mantiene i nomi delle feature
            model.predict_proba,
            num_features=7
        )

        # Estrai dati per il plot
        # 1) Probabilità
        probs = model.predict_proba(df_input.iloc[[i]])[0]
        df_prob = pd.DataFrame({
            'Classe': ['Not at risk','At risk'],
            'Probabilità': probs
        })

        # 2) Contributi
        feat_contrib = exp.as_list(label=1)  # lista di tuple (feature_expr, contrib)
        df_contrib = pd.DataFrame(feat_contrib, columns=['feature_expr','contrib'])
        df_contrib['color'] = df_contrib.contrib.apply(lambda x: 'orangered' if x>0 else 'steelblue')

        # --- qui la magia: ricavo il nome “pulito” della feature ---
        # es. "ST_Slope_Up <= 0.00" -> "ST_Slope_Up"
        df_contrib['feat_name'] = df_contrib['feature_expr'].str.split(' ', 1).str[0]
        df_contrib['feat_pretty'] = df_contrib['feat_name'].map(FEATURE_LABELS).fillna(df_contrib['feat_name'])

        # lista dei soli nomi da cercare nel DataFrame
        feat_names = df_contrib['feat_name'].tolist()

        # 🔍 Filtra solo quelli davvero presenti in df_input.columns
        valid_feat_names = [f for f in feat_names if f in df_input.columns]

        # 3) Valori: estraggo i valori originali per ciascuna feat_name valida
        df_vals = df_input.iloc[i][valid_feat_names].to_frame(name='Value')

        # Prepare & sort: positives first (largest → smallest), then negatives (closest to zero → most negative)
        shap_items = list(zip(single_exp.feature_names,
                            single_exp.values,
                            single_exp.data))
        shap_sorted = sorted(shap_items, key=lambda x: x[1], reverse=True)

        
        # Sort contributions (highest → lowest)
        df_lime_sorted = df_contrib.sort_values(by='contrib', ascending=False).reset_index(drop=True)
        # ── Interfaccia ──
        st.title("🧠 XAI Dashboard")   


        # ── Spiegazione unica SHAP+LIME ──
        # 1) Feature SHAP
        shap_inc = [FEATURE_LABELS.get(f, f) for f, impact, _ in shap_sorted if impact > 0]
        shap_dec = [FEATURE_LABELS.get(f, f) for f, impact, _ in shap_sorted if impact < 0]

        # 2) Feature LIME
        lime_inc = [
            FEATURE_LABELS.get(row.feat_name, row.feat_name)
            for _, row in df_lime_sorted.iterrows()
            if row.contrib > 0
        ]
        lime_dec = [
            FEATURE_LABELS.get(row.feat_name, row.feat_name)
            for _, row in df_lime_sorted.iterrows()
            if row.contrib < 0
        ]


        # 1) Definisco lo stile con supporto light/dark
        st.markdown("""
            <style>
            .explanation-box {
                border-left: 5px solid #2196F3;
                border-radius: 8px;
                padding: 18px 22px;
                margin: 20px 0;
                font-family: 'Segoe UI', Tahoma, sans-serif;
                line-height: 1.6em;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: background 0.3s, color 0.3s;
            }

            /* Modalità chiara */
            @media (prefers-color-scheme: light) {
            .explanation-box {
                background: #f0f4f8;
                color: #333;
            }
            .explanation-box h3 {
                color: #1a73e8;
            }
            .explanation-box strong {
                color: #1a73e8;
            }
            }

            /* Modalità scura */
            @media (prefers-color-scheme: dark) {
            .explanation-box {
                background: #2a2a2a;
                color: #cfcfcf;
            }
            .explanation-box h3 {
                color: #90caf9;
            }
            .explanation-box strong {
                color: #64b5f6;
            }
            }
            </style>
        """, unsafe_allow_html=True)

        # 2) Costruisco il contenuto in un div per applicare lo stile
        text_unico = (
            "<div class='explanation-box'>"
            "<h3>🔍 Quick Explanation</h3>"
            "<strong>1. First methodology</strong>: this provides an overall view by identifying which factors, on average, tend to push the risk up or down among all patients.<br><br>"
            f"• Factors that tend to increase risk: {', '.join(shap_inc)}.<br>"
            f"• Factors that tend to decrease risk: {', '.join(shap_dec)}.<br><br>"
            "<strong>2. Second methodology</strong>: this focuses on your individual case, showing which inputs had the strongest influence on your specific prediction.<br><br>"
            f"• Factors that increased your personal risk: {', '.join(lime_inc)}.<br>"
            f"• Factors that decreased your personal risk: {', '.join(lime_dec)}.<br><br>"
            f"Overall, based on these analyses, you are: "
            f"<b style=\"color: #64b5f6;\">{'at risk of heart failure' if pred == 1 else 'not at risk of heart failure'}</b></div>"
        )

        # 3) Stampo il box
        st.markdown(text_unico, unsafe_allow_html=True)


        # crea 2 tab: 0=SHAP, 1=LIME
        st.markdown("### Advanced interpretation", unsafe_allow_html=True)

        tab_shap, tab_lime = st.tabs(["SHAP Explanation", "LIME Explanation"])
        with tab_shap:
            #stampa la baseline in modo carino
            # Legend


            st.markdown("""
            <div style="display:flex; justify-content:center; gap:2rem; margin-bottom:1rem;">
                <span>
                    <span style="color:#E74C3C;">🔺</span>
                    <span style="color:#827f78; font-weight:bold;">Positive impact</span>
                </span>
                <span>
                    <span style="color:#3498DB;">🔻</span>
                    <span style="color:#827f78; font-weight:bold;">Negative impact</span>
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Textual interpretation in order
            for feature, impact, value in shap_sorted:
                if impact > 0:
                    st.markdown(
                        f"""
                        <span>
                            <span style='color:#827f78;'>🔺 <strong>{feature}</strong> - </span>
                            <span style='color:#E74C3C;'>increases risk by <strong>{impact:.2f}%</strong>.</span>
                        </span>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <span>
                            <span style='color:#827f78;'>🔻 <strong>{feature}</strong> - </span>
                            <span style='color:#3498DB;'>decreases risk by <strong>{abs(impact):.2f}%</strong>.</span>
                        </span>
                        """,
                        unsafe_allow_html=True
                    )
            st.markdown(
                f"""
                <div style='color:#827f78; margin-top:0.8%; margin-left:0.5%;'>
                    <b>Graphical interpretation</b>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Baseline
            st.markdown(
                f"""
                <div style='color:#827f78; margin-top:0.8%; margin-left:0.5%;'>
                    <u><b><i>Baseline -> </i></b>{base:.2f}%</u>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Then the force plot
            force_plot = shap.plots.force(
                explainer.expected_value[1] * 100,
                single_exp.values,
                single_exp.data,
                feature_names=single_exp.feature_names,
                matplotlib=False
            )
            st_shap(force_plot, height=300, width=900)


        with tab_lime:

            # 1️⃣ Textual interpretation first
            st.markdown(
                """
                    <div style="display:flex; justify-content:center; gap:2rem; margin-bottom:1rem;">
                            <span>
                                <span style="color:#E74C3C;">🔺</span>
                                <span style="color:#827f78; font-weight:bold;">Positive impact</span>
                            </span>
                            <span>
                                <span style="color:#3498DB;">🔻</span>
                                <span style="color:#827f78; font-weight:bold;">Negative impact</span>
                            </span>
                        
                    </div>
                """
            , unsafe_allow_html=True)
            for _, row in df_lime_sorted.iterrows():
                color = "#E74C3C" if row.contrib > 0 else "#3498DB"
                arrow = "🔺" if row.contrib > 0 else "🔻"
                verb = "increases" if row.contrib > 0 else "decreases"
                st.markdown(
                    f"""
                    <span>
                    <!-- Arrow + feature in #827f78 -->
                    <span style='color:#827f78;'>
                        {arrow} <strong>{row.feat_pretty} - </strong>
                    </span>
                    <!-- Numeric part in red/blue -->
                    <span style='color:{color};'>
                        {verb} risk by <strong>{abs(row.contrib):.2f}</strong> points.
                    </span>
                    </span>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown(
                f"""
                <div style='color:#827f78; margin-top:0.8%; margin-left:0.5%;'>
                    <b>Graphical interpretation</b>
                </div>
                """,
                unsafe_allow_html=True
            )
            # 2️⃣ Then the bar chart
            fig_c = go.Figure(go.Bar(
                x=df_lime_sorted['contrib'],
                y=df_lime_sorted['feat_pretty'],
                orientation='h',
                marker_color=df_lime_sorted['color']
            ))
            fig_c.update_layout(
                title="",
                xaxis_title="Contribution",
                yaxis_title="Feature",
                margin=dict(l=200, r=20, t=50, b=20)
            )
            st.plotly_chart(fig_c, use_container_width=True)

    if not st.session_state.predicted:
            st.markdown(
                """
                <div style="text-align:center; margin-top:10rem; color:#827f78;">
                    <h3>❤️‍🩹 Welcome to the Explainable Heart Failure Risk Prediction</h3>
                    <p>Select the patient parameters in the sidebar<br>
                    and click the <strong style="color:#2196F3;">Predict</strong> button to see the result!</p>
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    run()
