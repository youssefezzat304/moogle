import streamlit as st

from src.lunar_apps.dashboard.ui.mlm_training_dashboard import render_mlm_training_dashboard


st.set_page_config(
    page_title="LunarGeo MLM Training",
    layout="wide",
)

render_mlm_training_dashboard()
