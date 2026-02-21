import streamlit as st
import pandas as pd
import pickle
import kagglehub
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("churn_model.pkl", "rb"))

st.title("Customer Churn Prediction Dashboard")

# -----------------------------
# Load Dataset
# -----------------------------
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
df = pd.read_csv(f"{path}/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

encoder = LabelEncoder()
for column in df.select_dtypes(include="object").columns:
    df[column] = encoder.fit_transform(df[column])

X = df.drop("Churn", axis=1)
y = df["Churn"]

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Controls")

use_random = st.sidebar.button("🎲 Generate Random Customer")

# -----------------------------
# Input Data Logic
# -----------------------------
if use_random:
    input_data = X.sample(1)
    st.sidebar.success("Random customer generated")

else:
    st.sidebar.header("Customer Inputs")

    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges", 0, 200, 70)
    total_charges = tenure * monthly_charges

    contract = st.sidebar.selectbox("Contract", sorted(df["Contract"].unique()))
    internet_service = st.sidebar.selectbox("Internet Service", sorted(df["InternetService"].unique()))
    payment_method = st.sidebar.selectbox("Payment Method", sorted(df["PaymentMethod"].unique()))

    input_data = X.median().to_frame().T

    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges
    input_data["Contract"] = contract
    input_data["InternetService"] = internet_service
    input_data["PaymentMethod"] = payment_method

# -----------------------------
# Display Input
# -----------------------------
st.subheader("Model Input Vector")
st.write(input_data)

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)

st.subheader("Prediction")

if prediction[0] == 1:
    st.error("Customer Likely to Churn")
else:
    st.success("Customer Likely to Stay")

st.write("Probability:")
st.write(probability)

# -----------------------------
# Model Performance
# -----------------------------
st.subheader("Model Performance")

y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)

st.write("Accuracy:", accuracy)

cm = confusion_matrix(y, y_pred)

fig, ax = plt.subplots()
ax.matshow(cm)

plt.xlabel("Predicted")
plt.ylabel("Actual")

for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, val)

st.pyplot(fig)

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("Top Features")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.write(importance.head(10))