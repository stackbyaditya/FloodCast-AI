import streamlit as st

st.set_page_config(
    page_title="Flood AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌊 Flood AI")

st.subheader("AI Powered Flood Forecasting Platform")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Model", "XGBoost")
col2.metric("R²", "0.985")
col3.metric("RMSE", "25.12")
col4.metric("Dataset", "4548 Events")

st.markdown("---")

st.image(
    "https://images.unsplash.com/photo-1547683905-f686c993aae5?w=1400",
    use_container_width=True
)

st.markdown(
"""
## Welcome

Flood AI is an AI-powered flood prediction platform capable of predicting

- Peak Flood Level
- Batch Flood Prediction
- Model Explainability
- Interactive Analytics

using Machine Learning.
"""
)