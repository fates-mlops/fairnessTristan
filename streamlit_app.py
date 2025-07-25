import streamlit as st
from app import home, new_experiment, recap

PAGES = {
    "Home": home,
    "New Experiment": new_experiment,
    "Recap": recap,
}

st.sidebar.title("Menu")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]

page.app()