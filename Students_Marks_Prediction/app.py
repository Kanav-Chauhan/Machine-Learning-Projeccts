import streamlit as st
import pandas as pd
import pickle
import kagglehub
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("student_performance_model.pkl", "rb"))

st.title("Student Performance Prediction Dashboard")

st.write(
    "This model predicts a student's Math Score based on demographic "
    "attributes and academic indicators."
)

# -----------------------------
# Load Dataset
# -----------------------------
path = kagglehub.dataset_download("spscientist/students-performance-in-exams")
df = pd.read_csv(f"{path}/StudentsPerformance.csv")

# -----------------------------
# Model Description
# -----------------------------
st.subheader("Model Information")

st.write(
    """
**Model Type:** Random Forest Regressor  
**Problem Type:** Regression  
**Prediction Target:** Math Score  

Random Forest is an ensemble learning method that combines multiple decision
trees to improve prediction stability and reduce variance.
"""
)

# -----------------------------
# Random Button
# -----------------------------
st.sidebar.header("Controls")
random_student = st.sidebar.button("🎲 Generate Random Student")

# -----------------------------
# Input Logic
# -----------------------------
if random_student:
    original_input = df.sample(1)
    st.sidebar.success("Random student generated")

else:
    st.sidebar.header("Student Inputs")

    gender = st.sidebar.selectbox("Gender", df["gender"].unique())
    race = st.sidebar.selectbox("Race / Ethnicity", df["race/ethnicity"].unique())
    education = st.sidebar.selectbox("Parental Education", df["parental level of education"].unique())
    lunch = st.sidebar.selectbox("Lunch Type", df["lunch"].unique())
    prep = st.sidebar.selectbox("Test Preparation Course", df["test preparation course"].unique())

    reading_score = st.sidebar.slider("Reading Score", 0, 100, 50)
    writing_score = st.sidebar.slider("Writing Score", 0, 100, 50)

    original_input = pd.DataFrame({
        "gender": [gender],
        "race/ethnicity": [race],
        "parental level of education": [education],
        "lunch": [lunch],
        "test preparation course": [prep],
        "reading score": [reading_score],
        "writing score": [writing_score],
    })

# -----------------------------
# Show ORIGINAL Values
# -----------------------------
st.subheader("Student Profile")
st.write(original_input)

# -----------------------------
# Prepare Input for Model
# -----------------------------
input_data = original_input.copy()

if "math score" in input_data.columns:
    input_data = input_data.drop("math score", axis=1)

encoder = LabelEncoder()

for column in input_data.columns:
    if input_data[column].dtype == "object":
        encoder.fit(df[column])
        input_data[column] = encoder.transform(input_data[column])

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(input_data)

st.subheader("Predicted Math Score")
st.success(f"Estimated Math Score: {prediction[0]:.2f}")

# -----------------------------
# Model Performance Metrics
# -----------------------------
st.subheader("Model Performance")

# Encode full dataset for evaluation
df_eval = df.copy()

for column in df_eval.columns:
    if df_eval[column].dtype == "object":
        encoder.fit(df[column])
        df_eval[column] = encoder.transform(df_eval[column])

X_eval = df_eval.drop("math score", axis=1)
y_eval = df_eval["math score"]

y_pred_eval = model.predict(X_eval)

mae = mean_absolute_error(y_eval, y_pred_eval)
rmse = np.sqrt(mean_squared_error(y_eval, y_pred_eval))
r2 = r2_score(y_eval, y_pred_eval)

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("R² Score", f"{r2:.3f}")

st.write(
    """
**MAE (Mean Absolute Error):** Average prediction error magnitude  
**RMSE (Root Mean Squared Error):** Penalizes larger errors  
**R² Score:** Variance explained by the model
"""
)

# -----------------------------
# Feature Importance Graph
# -----------------------------
st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature": X_eval.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=True)

fig, ax = plt.subplots()
ax.barh(importance["Feature"], importance["Importance"])
ax.set_xlabel("Importance")

st.pyplot(fig)

# -----------------------------
# Score Distribution
# -----------------------------
st.subheader("Math Score Distribution")

fig2, ax2 = plt.subplots()
ax2.hist(df["math score"], bins=20)
ax2.set_xlabel("Math Score")
ax2.set_ylabel("Frequency")

st.pyplot(fig2)