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
import time
# Dictionary mapping feature names to human-readable labels for display
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

ONEHOT_GROUPS = {
    "Sex": ["Sex_F", "Sex_M"],
    "ChestPainType": ["ChestPainType_ASY", "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA"],
    "RestingECG": ["RestingECG_LVH", "RestingECG_Normal", "RestingECG_ST"],
    "ExerciseAngina": ["ExerciseAngina_N", "ExerciseAngina_Y"],
    "ST_Slope": ["ST_Slope_Down", "ST_Slope_Flat", "ST_Slope_Up"]
}


def st_shap(plot, height=None, width=None):
    """
    Integrates a SHAP force plot (matplotlib=False) into Streamlit.
    """
    shap_js   = shap.getjs()
    plot_html = plot.html()
    html      = f"<head>{shap_js}</head><body>{plot_html}</body>"
    components.html(html, height=height, width=width)

def run():
    # Set up Streamlit page configuration
    st.set_page_config(
        page_title="Explainable Heart Failure Risk Predictor",
        page_icon="❤️",
        layout="wide",   
    )
    # Initialize session state for prediction
    if 'predicted' not in st.session_state:
        st.session_state.predicted = False
    
    warnings.simplefilter("ignore", category=FutureWarning)

    # Cache model loading for performance
    @st.cache_data
    def load_model(path):
        return joblib.load(path)

    # Cache background data loading and preprocessing for SHAP/LIME
    @st.cache_data
    def load_background(csv_path, n=100):
        df_bg = pd.read_csv(csv_path).head(n)
        num_cols = ['Age','RestingBP','Cholesterol','MaxHR']
        # Standardize numerical columns
        df_bg[num_cols] = StandardScaler().fit_transform(df_bg[num_cols])
        cats = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
        enc_df = pd.DataFrame(index=df_bg.index)
        # One-hot encode categorical columns
        for col in cats:
            enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            mat = enc.fit_transform(df_bg[[col]])
            names = enc.get_feature_names_out([col])
            enc_df = pd.concat([enc_df, pd.DataFrame(mat, columns=names, index=df_bg.index)], axis=1)
        proc = pd.concat([df_bg.drop(columns=cats), enc_df], axis=1)

        # Ensure all expected columns are present
        expected = [
            'Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak',
            'Sex_F','Sex_M','ChestPainType_ASY','ChestPainType_ATA','ChestPainType_NAP','ChestPainType_TA',
            'RestingECG_LVH','RestingECG_Normal','RestingECG_ST',
            'ExerciseAngina_N','ExerciseAngina_Y','ST_Slope_Down','ST_Slope_Flat','ST_Slope_Up'
        ]
        for c in expected:
            if c not in proc: proc[c] = 0
        return proc[expected].values

    # Load model and background data
    model = load_model("model/heart_disease_model.pkl")
    background_data = load_background("model/heart.csv")

    st.title("Explainable Heart Failure Risk Prediction")
    st.sidebar.header("Patient Parameters")

    # Sidebar toggle for advanced/base mode
    with st.sidebar:
        advanced = tog.st_toggle_switch(
            label="Active Advanced mode",      
            key="mode_switch",          
            default_value=False,        
            label_after=False,          
            inactive_color="#D3D3D3",   
            active_color="#11567F",     
            track_color="#29B5E8"       
        )
    mode = "Advanced" if advanced else "Base"

    # Advanced mode: sliders for precise input
    if mode == "Advanced":
        age = st.sidebar.slider("Age", 18, 120, 50, help="Patient age in completed years")
        resting_bp = st.sidebar.slider("Resting BP (mm Hg)", 80, 200, 120, help="Resting systolic Blood Pressure")
        cholesterol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 400, 200, help="Serum total cholesterol")
        max_hr = st.sidebar.slider("Max HR", 60, 200, 150, help="Peak exercise heart rate")
        oldpeak = st.sidebar.slider(
            "ST Depression (Oldpeak)", 0.0, 4.0, 1.0, step=0.1,
            help="Magnitude of ST-segment depression"
        )
        sex = st.sidebar.selectbox("Sex", ["Male","Female"])
        fasting_bs = st.sidebar.selectbox("Fasting BS >120 mg/dl", ["Yes", "No"], help="Elevated fasting glucose")

        # Sidebar categorical inputs
        chest_pain = st.sidebar.selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
            help="Chest pain classification"
        )
        
        resting_ecg = st.sidebar.selectbox(
            "Resting ECG",
            ["Normal", "ST-T Abnormality", "LVH by Estes"],
            help="Resting ECG interpretation"
        )
        exercise_angina = st.sidebar.selectbox("Exercise Angina", ["Yes", "No"], help="Exercise-induced angina")
        st_slope = st.sidebar.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"], help="Slope of ST segment")
    else:
        # Base mode: only age, systolic & diastolic BP, pulse rate, and diabetes
        def midpoint(label, opts, help_text=None):
            """Show a selectbox with ranges and return the midpoint value."""
            sel = st.sidebar.selectbox(label, opts, help=help_text)
            low, high = map(float, sel.split("-"))
            return (low + high) / 2

        # 1) Age range in 10-year bands
        age = midpoint(
            "Age range (years)",
            ["20-29","30-39","40-49","50-59","60-69","70-79","80-89"],
            help_text="Select the range that includes the patient's age"
        )

        # 2) Systolic blood pressure range
        systolic_bp = midpoint(
            "Systolic BP range (mm Hg)",
            ["90-109","110-129","130-149","150-169","170-189","190-210"],
            help_text="Choose the systolic blood pressure measured at rest"
        )

        # 3) Diastolic blood pressure range
        diastolic_bp = midpoint(
            "Diastolic BP range (mm Hg)",
            ["50-59","60-69","70-79","80-89","90-99","100-110"],
            help_text="Choose the diastolic blood pressure measured at rest"
        )

        # 4) Resting heart rate (pulse) range
        pulse = midpoint(
            "Heart rate (bpm)",
            ["40-59","60-74","75-89","90-104","105-120","121-140"],
            help_text="Select resting pulse rate in beats per minute"
        )

        # 5) Diabetes indicator
        fasting_bs = st.sidebar.selectbox(
            "Diabetes (Fasting blood sugar >120 mg/dL)",
            ["No","Yes"],
            help="Select 'Yes' if fasting glucose exceeded 120 mg/dL"
        )


    # When the user clicks "Predict"
    if st.sidebar.button("Predict"):
        st.session_state.predicted = True
        with st.spinner("⏳ Predicting..."):
            time.sleep(2)

            if mode == "Advanced":
                # Build input DataFrame from user input
                df_input = pd.DataFrame([{
                    "Age":age, "Sex":sex, "ChestPainType":chest_pain,
                    "RestingBP":resting_bp, "Cholesterol":cholesterol,
                    "FastingBS":1 if fasting_bs=="Yes" else 0,
                    "RestingECG":resting_ecg, "MaxHR":max_hr,
                    "ExerciseAngina":exercise_angina, "Oldpeak":oldpeak,
                    "ST_Slope":st_slope
                }])
                num_cols = ['Age','RestingBP','Cholesterol','MaxHR']
                # Standardize numerical columns
                df_input[num_cols] = StandardScaler().fit_transform(df_input[num_cols])
                cats = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
                enc_df = pd.DataFrame()
                # One-hot encode categorical columns
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
                # Ensure all expected columns are present
                for c in expected:
                    if c not in df_input: df_input[c] = 0
                df_input = df_input[expected]

                # Make prediction and get probabilities
                pred  = model.predict(df_input)[0]
                proba = model.predict_proba(df_input)[0]

                # Display prediction result
                st.subheader("Prediction Result")
                # --- Custom prediction bar as in the image ---
                # Calculate values
                not_risk_pct = proba[0] * 100
                risk_pct = proba[1] * 100
                threshold = 50  # central threshold

                # Soglie parametriche
                VERY_HIGH = 90
                HIGH      = 70
                MODERATE  = 55
                BORDER    = 50
                MARGIN    = 5
                # Decide message and color
                if not_risk_pct >= VERY_HIGH:
                    msg        = "✅ Very high confidence: patient is not at risk"
                    box_color  = "#a5d6a7"
                    font_color = "#1b5e20"

                elif not_risk_pct >= HIGH:
                    msg        = "✅ Low risk: patient is likely safe"
                    box_color  = "#c8e6c9"
                    font_color = "#1b5e20"

                elif not_risk_pct > MODERATE:
                    msg        = "Slightly leaning towards no risk"
                    box_color  = "#e3f2fd"
                    font_color = "#1565c0"

                elif abs(not_risk_pct - BORDER) <= MARGIN:
                    msg        = "Probability near 50% — review further"
                    box_color  = "#ffe082"
                    font_color = "#333"

                elif risk_pct >= VERY_HIGH:
                    msg        = "⚠️ Very high confidence: patient is at risk"
                    box_color  = "#ef9a9a"
                    font_color = "#b71c1c"

                elif risk_pct >= HIGH:
                    msg        = "⚠️ High risk: patient is likely at risk"
                    box_color  = "#ffb3b3"
                    font_color = "#b71c1c"

                elif risk_pct > MODERATE:
                    msg        = "⚠️ Moderate risk: consider further tests"
                    box_color  = "#ffcc80"
                    font_color = "#e65100"

                else:
                    msg        = "⚠️ Balanced risk — use clinical judgment"
                    box_color  = "#ffe082"
                    font_color = "#333"

                # Show message in a rounded box
                st.markdown(
                    f"""
                    <div style="
                        background: {box_color};
                        color: {font_color};
                        border-radius: 12px;
                        border: 1.5px solid #bbb;
                        padding: 12px 18px;
                        margin-bottom: 18px;
                        font-size: 1.1rem;
                        font-weight: 500;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
                        display: inline-block;
                    ">
                        {msg}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Custom horizontal bar with Plotly
                fig = go.Figure()

                # Not at risk bar (left)
                fig.add_trace(go.Bar(
                    x=[not_risk_pct],
                    y=[''],
                    orientation='h',
                    marker=dict(color='#388e3c'),
                    name='Not a risk',
                    hoverinfo='skip',
                    width=0.5,
                    showlegend=False,
                    text=[f"{not_risk_pct:.1f} %"],
                    textposition='inside',
                    insidetextanchor='middle',
                    textfont=dict(color='white', size=18)
                ))

                # At risk bar (right)
                fig.add_trace(go.Bar(
                    x=[risk_pct],
                    y=[''],
                    orientation='h',
                    marker=dict(color='#b3541a'),
                    name='At risk',
                    hoverinfo='skip',
                    width=0.5,
                    showlegend=False,
                    text=[f"{risk_pct:.1f} %"],
                    textposition='inside',
                    insidetextanchor='middle',
                    textfont=dict(color='white', size=18)
                ))

                # Add threshold marker (vertical line at 50%)
                fig.add_shape(
                    type="line",
                    x0=threshold, x1=threshold,
                    y0=-0.4, y1=0.4,
                    line=dict(color="black", width=3),
                    xref='x', yref='y'
                )

                # Add labels above the bar
                fig.add_annotation(
                    x=not_risk_pct/2,
                    y=0.25,
                    text="",
                    showarrow=False,
                    font=dict(color="#1b5e20", size=16)
                )
                fig.add_annotation(
                    x=not_risk_pct + risk_pct/2,
                    y=0.25,
                    text="",
                    showarrow=False,
                    font=dict(color="#b3541a", size=16)
                )

                # Layout settings
                fig.update_layout(
                    barmode='stack',
                    height=90,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(
                        range=[0, 100],
                        showticklabels=False,
                        showgrid=False,
                        zeroline=False,
                        fixedrange=True
                    ),
                    yaxis=dict(
                        showticklabels=False,
                        showgrid=False,
                        zeroline=False,
                        fixedrange=True
                    ),
                    plot_bgcolor='#e3f2fd',
                    paper_bgcolor='#e3f2fd',
                )

                st.plotly_chart(fig, use_container_width=True)

                # SHAP explanation
                explainer   = shap.Explainer(model, background_data)
                sv          = explainer(df_input)
                vals  = sv.values[0, :, 1] * 100    # SHAP values for class 1 (risk)
                base  = sv.base_values[0, 1] * 100
                data  = sv.data[0]
                names = sv.feature_names

                # Sort features by absolute SHAP value
                idx = np.argsort(np.abs(vals))[::-1]
                vals_top  = vals[idx]
                data_top  = data[idx]
                names_top = [names[i] for i in idx]

                # Map feature names to pretty labels
                pretty = []
                for feat in names_top:
                    pretty.append(FEATURE_LABELS.get(feat, feat))
                single_exp = Explanation(
                    values=vals_top,
                    base_values=base,
                    data=data_top,
                    feature_names=pretty
                )

                # LIME explanation
                lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=background_data,
                    feature_names=sv.feature_names,
                    class_names=['Not at risk', 'At risk'],
                    mode='classification'
                )

                i = 0
                exp = lime_explainer.explain_instance(
                    df_input.iloc[i],            
                    model.predict_proba,
                    num_features=7
                )

                # Prepare LIME output for display
                probs = model.predict_proba(df_input.iloc[[i]])[0]
                df_prob = pd.DataFrame({
                    'Classe': ['Not at risk','At risk'],
                    'Probabilità': probs
                })

                feat_contrib = exp.as_list(label=1)
                df_contrib = pd.DataFrame(feat_contrib, columns=['feature_expr','contrib'])
                df_contrib['color'] = df_contrib.contrib.apply(lambda x: 'orangered' if x>0 else 'steelblue')

                df_contrib['feat_name'] = df_contrib['feature_expr'].str.split(' ', 1).str[0]
                df_contrib['feat_pretty'] = df_contrib['feat_name'].map(FEATURE_LABELS).fillna(df_contrib['feat_name'])

                feat_names = df_contrib['feat_name'].tolist()

                valid_feat_names = [f for f in feat_names if f in df_input.columns]

                df_vals = df_input.iloc[i][valid_feat_names].to_frame(name='Value')

                # Prepare SHAP and LIME results for display

                # --- SHAP feature consolidation (grouping one-hot categories) ---
                adjusted_vals = {}
                used_features = set()
                for feat, impact, val in zip(names, vals, data):
                    if feat in used_features:
                        continue
                    for group, cols in ONEHOT_GROUPS.items():
                        if feat in cols:
                            active_col = next((c for c in cols if df_input.iloc[0][c] == 1), None)
                            total_contrib = sum(vals[names.index(c)] for c in cols if c in names)
                            label = f"{group} = {active_col.split('_')[-1]}" if active_col else group
                            adjusted_vals[group] = (label, total_contrib, 1)  # 1 is dummy value for visual
                            used_features.update(cols)
                            break
                    else:
                        adjusted_vals[feat] = (FEATURE_LABELS.get(feat, feat), impact, val)
                        used_features.add(feat)
                shap_sorted = sorted(adjusted_vals.values(), key=lambda x: abs(x[1]), reverse=True)

                df_lime_sorted = df_contrib.sort_values(by='contrib', ascending=False).reset_index(drop=True)

                # Remove that with name "0.00"
                df_lime_sorted = df_lime_sorted[df_lime_sorted['feat_name'] != "0.00"]
                df_lime_sorted['feat_pretty'] = df_lime_sorted['feat_name'].map(FEATURE_LABELS).fillna(df_lime_sorted['feat_name'])
                

                st.title("🧠 XAI Dashboard")   

                # Lists of features increasing/decreasing risk for SHAP and LIME
                shap_inc = [FEATURE_LABELS.get(f, f) for f, impact, _ in shap_sorted if impact > 0]
                shap_dec = [FEATURE_LABELS.get(f, f) for f, impact, _ in shap_sorted if impact < 0]

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

                # Custom CSS for explanation box
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

                    /* Light mode */
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

                    /* Dark mode */
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

                # Tabs: Quick Explanation, SHAP, LIME
                tab_quick, tab_shap, tab_lime = st.tabs(["Summary", "SHAP Explanation", "LIME Explanation"])

                with tab_quick:
                    text_unico = (
                        "<div class='explanation-box'>"
                        "<ol>"
                            "<li>"
                            "<strong>SHAP methodology</strong>: provides an overall view by identifying which factors, on average, tend to push the risk up or down among all patients."
                            "<ul>"
                                f"<li>Factors that tend to increase risk: {', '.join(shap_inc) if shap_inc else 'None'}.</li>"
                                f"<li>Factors that tend to decrease risk: {', '.join(shap_dec) if shap_dec else 'None'}.</li>"
                            "</ul>"
                            "</li>"
                            "<li>"
                            "<strong>LIME methodology</strong>: focuses on your individual case, showing which inputs had the strongest influence on your specific prediction."
                            "<ul>"
                                f"<li>Factors that increased your personal risk: {', '.join(lime_inc) if lime_inc else 'None'}.</li>"
                                f"<li>Factors that decreased your personal risk: {', '.join(lime_dec) if lime_dec else 'None'}.</li>"
                            "</ul>"
                            "</li>"
                        "</ol>"
                        f"<p><strong>Overall</strong>, based on these analyses, you are: "
                        f"<b style='color: #64b5f6;'>"
                            f"{'at risk of heart failure' if pred == 1 else 'not at risk of heart failure'}"
                        "</b></p>"
                        "</div>"
                    )
                    st.markdown(text_unico, unsafe_allow_html=True)

                with tab_shap:
                    # Legend for SHAP impacts
                    st.markdown("""
                    <div style="display:flex; justify-content:center; gap:2rem; margin-bottom:1rem;">
                                    <span>
                                        <span style="color:#E74C3C;">&#x25B2</span>
                                        <span style="color:#827f78; font-weight:bold;">Positive impact</span>
                                    </span>
                                    <span>
                                        <span style="color:#3498DB;">&#x25BC;</span>
                                        <span style="color:#827f78; font-weight:bold;">Negative impact</span>
                                    </span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(
                        f"""
                        <div style='color:#827f78; margin-top:0.8%; margin-left:0.5%;'>
                            <b>Graphical interpretation</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"""
                        <div style='color:#827f78; margin-top:3%; margin-left:0.5%;'>
                            <u><b><i>Baseline:</i></b>{base:.2f}%</u>
                        </div>
                        """,
                        unsafe_allow_html=True, help="Predicted risk without any new feature impact"
                    )

                    # SHAP force plot
                    force_plot = shap.plots.force(
                        explainer.expected_value[1] * 100,
                        single_exp.values,
                        single_exp.data,
                        feature_names=single_exp.feature_names,
                        matplotlib=False
                    )
                    st_shap(force_plot, height=320, width=900)

                    st.markdown(
                        """
                        <div style="display: flex; justify-content: flex-start; gap: 3rem; background: #e3f2fd; border-radius: 8px; padding: 10px 18px; margin-top: -10rem;">
                            <div>
                        """
                        +
                        "".join(
                            f"""<div style="margin-bottom:2px;">
                                <span style="color:#1565c0;font-weight:bold;">&#x25BC; <b>{feature}</b></span>
                                <span style="color:#222;">– decreases risk by </span>
                                <span style="color:#1565c0;font-weight:bold;">{abs(impact):.2f}%</span>
                            </div>"""
                            for feature, impact, _ in shap_sorted if impact < 0
                        )
                        +
                        "</div><div>"
                        +
                        "".join(
                            f"""<div style="margin-bottom:2px;">
                                <span style="color:#c62828;font-weight:bold;">&#x25B2; <b>{feature}</b></span>
                                <span style="color:#222;">– increases risk by </span>
                                <span style="color:#c62828;font-weight:bold;">{impact:.2f}%</span>
                            </div>"""
                            for feature, impact, _ in shap_sorted if impact > 0
                        )
                        +
                        "</div></div>"
                        ,
                        unsafe_allow_html=True
                    )

                with tab_lime:
                    # Legend for LIME impacts
                    st.markdown(
                        """
                            <div style="display:flex; justify-content:center; gap:2rem; margin-bottom:1rem;margin-top:0.1rem;">
                                    <span>
                                        <span style="color:#E74C3C;">&#x25B2</span>
                                        <span style="color:#827f78; font-weight:bold;">Positive impact</span>
                                    </span>
                                    <span>
                                        <span style="color:#3498DB;">&#x25BC;</span>
                                        <span style="color:#827f78; font-weight:bold;">Negative impact</span>
                                    </span>
                                
                            </div>
                        """
                    , unsafe_allow_html=True)
                    # List LIME feature impacts (old, now moved below)
                    # for _, row in df_lime_sorted.iterrows():
                    #     color = "#E74C3C" if row.contrib > 0 else "#3498DB"
                    #     arrow = "🔺" if row.contrib > 0 else "🔻"
                    #     verb = "increases" if row.contrib > 0 else "decreases"
                    #     st.markdown(
                    #         f"""
                    #         <span>
                    #         <!-- Arrow + feature in #827f78 -->
                    #         <span style='color:#827f78;'>
                    #             {arrow} <strong>{row.feat_pretty} - </strong>
                    #         </span>
                    #         <!-- Numeric part in red/blue -->
                    #         <span style='color:{color};'>
                    #             {verb} risk by <strong>{abs(row.contrib):.2f}</strong> points.
                    #         </span>
                    #         </span>
                    #         """,
                    #         unsafe_allow_html=True
                    #     )
                    st.markdown(
                        f"""
                        <div style='color:#827f78; margin-top:0.8%; margin-left:0.5%;'>
                            <b>Graphical interpretation</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # LIME bar plot
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

                    # --- LIME factors summary as in screenshot ---
                    st.markdown(
                        """
                        <div style="display: flex; justify-content: flex-start; gap: 3rem; background: #e3f2fd; border-radius: 8px; padding: 10px 18px; margin-top: 18px;">
                            <div>
                        """
                        +
                        "".join(
                            f"""<div style="margin-bottom:2px;">
                                <span style="color:#1565c0;font-weight:bold;">&#x25BC; <b>{row.feat_pretty}</b></span>
                                <span style="color:#222;">– decreases risk by </span>
                                <span style="color:#1565c0;font-weight:bold;">{abs(row.contrib):.2f}</span>
                            </div>"""
                            for _, row in df_lime_sorted.iterrows() if row.contrib < 0
                        )
                        +
                        "</div><div>"
                        +
                        "".join(
                            f"""<div style="margin-bottom:2px;">
                                <span style="color:#c62828;font-weight:bold;">&#x25B2; <b>{row.feat_pretty}</b></span>
                                <span style="color:#222;">– increases risk by </span>
                                <span style="color:#c62828;font-weight:bold;">{row.contrib:.2f}</span>
                            </div>"""
                            for _, row in df_lime_sorted.iterrows() if row.contrib > 0
                        )
                        +
                        "</div></div>"
                        ,
                        unsafe_allow_html=True
                    )
            else:  # esci dopo il flusso Advanced
                # --- Age points ---
                age_table = [
                    (20, 29, 0), (30, 39, 5), (40, 49, 10), (50, 59, 15),
                    (60, 69, 20), (70, 79, 25), (80, 89, 30)
                ]
                def get_age_points(a):
                    for lo, hi, pts in age_table:
                        if lo <= a <= hi:
                            return pts
                    return 0

                # --- Blood Pressure Category and Points (from table) ---
                def bp_category(sbp, dbp):
                    if sbp > 180 or dbp > 110:
                        return "Hypertensive Crisis"
                    elif sbp >= 160 or dbp >= 100:
                        return "Hypertension Stage 2"
                    elif sbp >= 140 or dbp >= 90:
                        return "Hypertension Stage 1"
                    elif sbp >= 120 or dbp >= 80:
                        return "Prehypertension"
                    else:
                        return "Normal"

                def bp_points(cat):
                    # More points for higher risk (red), less for green
                    return {
                        "Normal": 0,
                        "Prehypertension": 5,
                        "Hypertension Stage 1": 15,
                        "Hypertension Stage 2": 25,
                        "Hypertensive Crisis": 40
                    }[cat]

                # --- Pulse rate points (unchanged) ---
                pulse_table = [
                    (0, 59,   5), (60, 74,  0), (75, 89,   5),
                    (90,104, 10), (105,120,15), (121,1000,20)
                ]
                def get_pulse_points(p):
                    for lo, hi, pts in pulse_table:
                        if lo <= p <= hi:
                            return pts
                    return 0

                # --- Diabetes points (unchanged) ---
                def get_diab_points(f):
                    return 12 if f == "Yes" else 0

                # --- Calculate BP category and points ---
                bp_cat = bp_category(systolic_bp, diastolic_bp)
                bp_pts = bp_points(bp_cat)

                # --- Sum all points ---
                total = (
                    get_age_points(age)
                    + bp_pts
                    + get_pulse_points(pulse)
                    + get_diab_points(fasting_bs)
                )

                # --- Map total points to risk: green (low), yellow (balanced), red (high) ---
                # Aggressive mapping: green (0-9), yellow (10-24), orange (25-39), red (40+)
                if total <= 9:
                    risk_pct = "5%"
                    level = "Low risk"
                elif total <= 24:
                    risk_pct = "15%"
                    level = "Balanced risk"
                elif total <= 39:
                    risk_pct = "35%"
                    level = "Intermediate risk"
                else:
                    risk_pct = "60%"
                    level = "High risk"

                num_risk = float(risk_pct.strip("%<>"))
                not_risk_pct = 100 - num_risk

                # Thresholds and messages: green = not at risk, red = at risk, yellow = balanced
                if num_risk >= 50:
                    msg, box_color, font_color = "⚠️ Very high risk: immediate attention needed", "#ef9a9a", "#b71c1c"
                elif num_risk >= 35:
                    msg, box_color, font_color = "⚠️ High risk: likely at risk", "#ffb3b3", "#b71c1c"
                elif num_risk >= 15:
                    msg, box_color, font_color = "⚠️ Balanced risk — monitor closely", "#ffe082", "#333"
                else:
                    msg, box_color, font_color = "✅ Low risk: patient is likely safe", "#c8e6c9", "#1b5e20"

                # Show message in a rounded box
                st.markdown(f"""
                    <div style="
                        background: {box_color};
                        color: {font_color};
                        border-radius: 12px;
                        border: 1.5px solid #bbb;
                        padding: 12px 18px;
                        margin-bottom: 18px;
                        font-size: 1.1rem;
                        font-weight: 500;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
                        display: inline-block;
                    ">
                        {msg}
                    </div>
                """, unsafe_allow_html=True)

                # Custom horizontal bar with Plotly
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[not_risk_pct], y=[''], orientation='h',
                    marker=dict(color='#388e3c'), showlegend=False,
                    text=[f"{not_risk_pct:.1f} %"], textposition='inside', insidetextanchor='middle',
                    textfont=dict(color='white', size=18)
                ))
                fig.add_trace(go.Bar(
                    x=[num_risk], y=[''], orientation='h',
                    marker=dict(color='#b3541a'), showlegend=False,
                    text=[f"{num_risk:.1f} %"], textposition='inside', insidetextanchor='middle',
                    textfont=dict(color='white', size=18)
                ))
                fig.add_shape(type="line", x0=50, x1=50, y0=-0.4, y1=0.4,
                            line=dict(color="black", width=3), xref='x', yref='y')
                fig.update_layout(
                    barmode='stack', height=90, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(range=[0,100], showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    plot_bgcolor='#e3f2fd', paper_bgcolor='#e3f2fd'
                )
                st.plotly_chart(fig, use_container_width=True)

                # 🧠 XAI Dashboard – Summary
                st.title("🧠 XAI Dashboard")

                # Note sulle anomalie
                abnormal = []
                if get_age_points(age) >= 10:
                    abnormal.append(f"Age: **{age} years** (→ {get_age_points(age)} points)")
                if bp_pts > 0:
                    abnormal.append(f"Blood Pressure: **{systolic_bp}/{diastolic_bp} mm Hg** ({bp_cat}, → {bp_pts} points)")
                if get_pulse_points(pulse) > 0:
                    abnormal.append(f"Pulse rate: **{pulse} bpm** (→ {get_pulse_points(pulse)} points)")
                if fasting_bs == "Yes":
                    abnormal.append("Diabetes: **Yes** (→ 12 points)")

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"""
                        <div style="
                            background:#e0e5ec;
                            padding:16px;
                            border-radius:8px;
                            text-align:center;
                            color:#222;
                        ">
                            <h3 style="margin:0; color:#1565c0;">{risk_pct}</h3>
                            <p style="font-weight:600; color:#444;">{level}</p>
                        </div>
                        """, unsafe_allow_html=True)

                with col2:
                    # prepare HTML legend
                    legend_html = """
                    <div style="margin-bottom:8px; font-size:0.9rem;">
                        <span style="color:#43a047; font-size:1.2rem;">&#9679;</span> Good value
                        &nbsp;&nbsp;
                        <span style="color:#e53935; font-size:1.2rem;">&#9679;</span> Review value
                    </div>
                    """

                    # Funzione di supporto per il dot colorato
                    def dot_html(is_abnormal):
                        color = "#e53935" if is_abnormal else "#43a047"
                        return f"<span style='color:{color}; font-size:1.1rem;'>&#9679;</span>"

                    # Verifica anomalie per ciascun campo
                    age_bad   = get_age_points(age) >= 10
                    bp_bad    = bp_pts > 0
                    pulse_bad = get_pulse_points(pulse) > 0
                    diab_bad  = fasting_bs == "Yes"

                    # Costruisce la lista HTML
                    list_items = f"""
                    <li>{dot_html(age_bad)} Age: {age} years</li>
                    <li>{dot_html(bp_bad)} Blood Pressure: {systolic_bp}/{diastolic_bp} mm Hg ({bp_cat})</li>
                    <li>{dot_html(pulse_bad)} Pulse rate: {pulse} bpm</li>
                    <li>{dot_html(diab_bad)} Diabetes: {'Yes' if diab_bad else 'No'}</li>
                    """

                    html_content = f"""
                    <div style="
                        padding: 12px 16px;
                        background: #e0e5ec;
                        border-radius: 8px;
                        color: #222;
                    ">
                        {legend_html}
                        <ul style="
                            margin: 0;
                            padding-left: 1.2rem;
                            font-size: 0.95rem;
                        ">
                            {list_items}
                        </ul>
                    </div>
                    """

                    # Renderizza l’HTML tramite il componente
                    components.html(html_content, height=200)
# If no prediction yet, show welcome message
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

# Entry point
if __name__ == "__main__":
    run()