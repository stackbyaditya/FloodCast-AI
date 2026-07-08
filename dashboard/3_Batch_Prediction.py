import streamlit as st
import pandas as pd

st.title("📂 Batch Prediction")

uploaded = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded:

    df = pd.read_csv(uploaded)

    st.dataframe(df.head())