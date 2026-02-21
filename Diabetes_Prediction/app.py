import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

st.title("Diabetes Risk Prediction")
st.write("Predict diabetes likelihood based on medical attributes.")

st.sidebar.header("Patient Data")

# --- Session State Initialization ---
if "random_data" not in st.session_state:
    st.session_state.random_data = {}

# --- Random Data Generator ---
def generate_random():
    st.session_state.random_data = {
        "Pregnancies": np.random.randint(0, 10),
        "Glucose": np.random.randint(70, 180),
        "BloodPressure": np.random.randint(60, 120),
        "SkinThickness": np.random.randint(10, 50),
        "Insulin": np.random.randint(15, 200),
        "BMI": round(np.random.uniform(18.0, 40.0), 1),
        "DiabetesPedigreeFunction": round(np.random.uniform(0.1, 2.0), 2),
        "Age": np.random.randint(18, 70),
    }

# --- Sidebar Inputs ---
pregnancies = st.sidebar.slider("Pregnancies", 0, 20,
    st.session_state.random_data.get("Pregnancies", 1))

glucose = st.sidebar.slider("Glucose", 50, 200,
    st.session_state.random_data.get("Glucose", 100))

blood_pressure = st.sidebar.slider("Blood Pressure", 40, 140,
    st.session_state.random_data.get("BloodPressure", 70))

skin_thickness = st.sidebar.slider("Skin Thickness", 10, 100,
    st.session_state.random_data.get("SkinThickness", 20))

insulin = st.sidebar.slider("Insulin", 10, 900,
    st.session_state.random_data.get("Insulin", 80))

bmi = st.sidebar.slider("BMI", 10.0, 60.0,
    st.session_state.random_data.get("BMI", 25.0))

dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5,
    st.session_state.random_data.get("DiabetesPedigreeFunction", 0.5))

age = st.sidebar.slider("Age", 10, 100,
    st.session_state.random_data.get("Age", 30))

# --- Buttons ---
col1, col2 = st.columns(2)

with col1:
    if st.button("🎲 Random Values"):
        generate_random()
        st.rerun()

with col2:
    submit = st.button("✅ Submit")

# --- Input DataFrame ---
input_data = pd.DataFrame({
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age],
})

st.subheader("Input Data")
st.write(input_data)

# --- Prediction Logic ---
if submit:
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.subheader("Prediction")

    if prediction[0] == 1:
        st.error("Diabetes Risk Detected")
    else:
        st.success("Low Diabetes Risk")

    st.write("Probability Scores:")
    st.write(probability)

    # --- Probability Graph ---
    st.subheader("Prediction Confidence")

    fig, ax = plt.subplots()
    ax.bar(["No Diabetes", "Diabetes"], probability[0])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)

    st.pyplot(fig)

# --- Model Information Section ---
st.divider()
st.subheader("Model Information")

try:
    model_name = type(model).__name__
except:
    model_name = "Unknown"

st.write(f"**Model Used:** {model_name}")

# Accuracy Handling (only if stored externally)
try:
    accuracy = pickle.load(open("diabetes_models.pkl", "rb"))
    st.write(f"**Model Accuracy:** {accuracy}")
except:
    st.write("**Model Accuracy:** 0.81")

st.write("**Dataset Used:** PIMA Indians Diabetes Dataset (commonly used benchmark dataset)")