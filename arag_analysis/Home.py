"""
ARAG Claims Analytics
=====================
Main entry point for the Streamlit application.
"""

import streamlit as st

st.set_page_config(
    page_title="Legal Claims Analytics",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 4rem 2rem; border-radius: 20px; margin-bottom: 2.5rem;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); text-align: center;'>
    <h1 style='color: white; margin: 0 0 1rem 0; font-size: 3rem; font-weight: 800; border: none;'>
        Legal Claims Analytics
    </h1>
    <p style='color: rgba(255,255,255,0.95); font-size: 1.3rem; margin: 0; font-weight: 300;'>
        Legal Expense Insurance Claims Analysis & Predictive Modeling
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### Select a page from the sidebar to get started:

**1. Data Analysis** - Exploratory analysis of claims portfolio, cost drivers, and litigator performance

**2. Predictive Modeling** - Machine learning models for demand forecasting, cost prediction, resolution time, and claim segmentation
""")
