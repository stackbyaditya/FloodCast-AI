import streamlit as st

st.title("🏠 Home")

st.write("Welcome to Flood AI.")

st.markdown("---")

st.header("Project Overview")

st.write("""
Flood AI predicts peak flood level using
an optimized XGBoost model trained on

- Hydrological Features
- Climatic Features
- Basin Characteristics
- Rainfall Statistics
- Socio-economic Indicators
""")