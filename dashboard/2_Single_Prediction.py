import streamlit as st
import random

st.title("🔮 Flood Prediction")

st.subheader("Enter Flood Parameters")

drainage = st.number_input(
    "Drainage Area",
    value=1000.0
)

rainfall = st.number_input(
    "Annual Rainfall",
    value=1200.0
)

population = st.number_input(
    "Population Density",
    value=500.0
)

if st.button("Predict"):

    prediction = random.uniform(40,60)

    st.success(
        f"Predicted Peak Flood Level : {prediction:.2f} m"
    )