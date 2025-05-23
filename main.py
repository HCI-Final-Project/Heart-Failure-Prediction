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
    else:
        # Base mode: select ranges, use midpoint as input
        def range_midpoint(label, options, help_text):
            """Helper: selectbox for ranges, returns midpoint"""
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

    # Sidebar categorical inputs
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

    # When the user clicks "Predict"
    if st.sidebar.button("Predict"):
        st.session_state.predicted = True
        with st.spinner("⏳ Predicting..."):
            time.sleep(2)
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

            # Decide message and color
            if abs(not_risk_pct - risk_pct) < 15:
                msg = "⚠️ It seems healthy but it looks is on risk, review it"
                box_color = "#ffe082"
                font_color = "#333"
            elif pred == 1:
                msg = "⚠️ Patient is at risk of heart failure"
                box_color = "#ffb3b3"
                font_color = "#b71c1c"
            else:
                msg = "✅ Patient is not at risk"
                box_color = "#c8e6c9"
                font_color = "#1b5e20"

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
            shap_items = list(zip(single_exp.feature_names,
                                single_exp.values,
                                single_exp.data))
            shap_sorted = sorted(shap_items, key=lambda x: x[1], reverse=True)

            df_lime_sorted = df_contrib.sort_values(by='contrib', ascending=False).reset_index(drop=True)

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
                        <span style="color:#E74C3C;">🔺</span>
                        <span style="color:#827f78; font-weight:bold;">Positive impact</span>
                    </span>
                    <span>
                        <span style="color:#3498DB;">🔻</span>
                        <span style="color:#827f78; font-weight:bold;">Negative impact</span>
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                # List SHAP feature impacts (old, now moved below)
                # for feature, impact, value in shap_sorted:
                #     if impact > 0:
                #         st.markdown(
                #             f"""
                #             <span>
                #                 <span style='color:#827f78;'>🔺 <strong>{feature}</strong> - </span>
                #                 <span style='color:#E74C3C;'>increases risk by <strong>{impact:.2f}%</strong>.</span>
                #             </span>
                #             """,
                #             unsafe_allow_html=True
                #         )
                #     else:
                #         st.markdown(
                #             f"""
                #             <span>
                #                 <span style='color:#827f78;'>🔻 <strong>{feature}</strong> - </span>
                #                 <span style='color:#3498DB;'>decreases risk by <strong>{abs(impact):.2f}%</strong>.</span>
                #             </span>
                #             """,
                #             unsafe_allow_html=True
                #         )
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
                            <span style="color:#1565c0;font-weight:bold;">{abs(impact):.0f}%</span>
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
                            <span style="color:#c62828;font-weight:bold;">{impact:.0f}%</span>
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
                            <span style="color:#1565c0;font-weight:bold;">{abs(row.contrib):.0f}</span>
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
                            <span style="color:#c62828;font-weight:bold;">{row.contrib:.0f}</span>
                        </div>"""
                        for _, row in df_lime_sorted.iterrows() if row.contrib > 0
                    )
                    +
                    "</div></div>"
                    ,
                    unsafe_allow_html=True
                )
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