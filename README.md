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

The app is designed with **Explainable AI (XAI)** in mind. Techniques such as `SHAP`, `LIME`, or the display of feature importance can be integrated (planned for future development).

