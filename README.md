# Heart Failure Risk Prediction Dashboard

## Developed by

 - Christian Ortega
 - Gioele Modica  
 - Giovanni Criscione  

## Project Goal

This project provides an **interactive dashboard** built with **Streamlit**, where users can input their clinical parameters and receive a **real-time prediction** of their risk of heart failure.

The app uses a machine learning model trained on a real-world dataset and is designed to be **interpretable and transparent** through **Explainable AI (XAI)** techniques.

---

## Key Features

- Input of patient clinical data through an interactive sidebar.
- Risk prediction (low or high) using a pre-trained ML model.
- Display of predicted probabilities.
- Support of XAI techniques to interpret the model's predictions.

---

## Project Structure

```plaintext
├── main.py                                 # Streamlit app
├── model/
│   └── heart_disease_model.pkl             # Serialized ML model (required for the dashboard)
|   └── heart-failure-prediction.ipynb      # EDA, preprocessing, model training and evaluation
|   └── heart.csv                           # Dataset
├── requirements.txt                        # Requirements for pip installation
└── environment.yml                         # Conda environment definition
```

---

## Environment Setup

You can install the dependencies using one of the following methods:

### Option 1 - Conda (recommended)

```bash
conda env create -f environment.yml
conda activate HCI_heart_attack
```

### Option 2 - pip

```bash
pip install -r requirements.txt
```

Make sure Python 3.9 or newer is installed.

---

## Running the Application

Once dependencies are installed and the `model/heart_disease_model.pkl` file is in place, you can launch the dashboard with:

```bash
streamlit run main.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## Input Parameters

Users can customize the following clinical parameters via the sidebar:

- Age  
- Sex  
- Chest pain type  
- Resting blood pressure  
- Cholesterol  
- Fasting blood sugar (>120 mg/dl)  
- Resting ECG results  
- Maximum heart rate (MaxHR)  
- Exercise-induced angina  
- ST depression (oldpeak)  
- ST slope  

---

## Model Details and Explainable AI

The classification model used (**Random Forest**) was trained using `scikit-learn` with the following techniques:

- **Standardization** of numerical features  
- **One-Hot Encoding** for categorical features  
- **Outlier removal** using z-score  
- **Cross-validation** and hyperparameter tuning  

---

## Explainable AI (XAI) in the Dashboard

This dashboard integrates two state-of-the-art XAI techniques to help users and clinicians understand the model's predictions:

### SHAP (SHapley Additive exPlanations)
- **Global and local explanations:** SHAP values show how each feature contributes to increasing or decreasing the risk, both for the individual prediction and across the dataset.
- **Force plot visualization:** The app displays a SHAP force plot, highlighting the most influential features for the current prediction.
- **Feature impact listing:** Features are listed in order of their impact, with clear indications of whether they increase or decrease the predicted risk.

### LIME (Local Interpretable Model-agnostic Explanations)
- **Instance-level explanations:** LIME explains the prediction for the specific patient by approximating the model locally with an interpretable model.
- **Bar chart visualization:** The dashboard shows a bar chart of the top features that most increased or decreased the risk for the current input.
- **Textual summary:** The app provides a textual summary of which features contributed most to the prediction.


### Acknowledgment
We used artificial intelligence tools as support for writing code and documentation.
 - [1] OpenAI. ChatGPT: Generative Pre-trained Transformer (versione GPT-4), 2023. Disponibile all’indirizzo: https://chat.openai.com.