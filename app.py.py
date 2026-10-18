
import streamlit as st
import pandas as pd
import joblib

# Load the saved ML model
model = joblib.load("machine_fault_model.pkl")
scaler = joblib.load("machine_fault_scaler.pkl")

# Page configuration
st.set_page_config(
    page_title="Machine Fault Prediction",
    page_icon="⚙️",
    layout="centered"
)

# Application title
st.title("⚙️ Machine Fault Detection")
st.write(
    "Enter machine sensor readings to predict "
    "whether the machine is normal or faulty."
)

st.info(
    "Demonstration project using a synthetic dataset. "
    "Predictions are not validated for real industrial use."
)

# Sensor inputs
temperature = st.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=200.0,
    value=65.0
)

voltage = st.number_input(
    "Voltage (V)",
    min_value=0.0,
    max_value=500.0,
    value=230.0
)

current = st.number_input(
    "Current (A)",
    min_value=0.0,
    max_value=100.0,
    value=6.0
)

vibration = st.number_input(
    "Vibration",
    min_value=0.0,
    max_value=20.0,
    value=1.2
)

speed = st.number_input(
    "Speed (RPM)",
    min_value=0.0,
    max_value=5000.0,
    value=1200.0
)

# Prediction button
if st.button("Predict Machine Status", type="primary"):

    input_data = pd.DataFrame([{
        "Temperature": temperature,
        "Voltage": voltage,
        "Current": current,
        "Vibration": vibration,
        "Speed": speed
    }])

    # Logistic Regression requires scaled data
    if isinstance(model, type(joblib.load("machine_fault_model.pkl"))):
        # The saved model type is handled below using its class name.
        pass

    if model.__class__.__name__ == "LogisticRegression":
        model_input = scaler.transform(input_data)
    else:
        model_input = input_data

    prediction = model.predict(model_input)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Fault Detected")
    else:
        st.success("✅ Machine Classified as Normal")

    # Display the entered values
    st.subheader("Sensor Readings")
    st.dataframe(input_data, use_container_width=True)
