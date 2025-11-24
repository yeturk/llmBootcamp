import streamlit as st
import requests
import os

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")

st.title("GTU Staj Chatbot - Streamlit Arayüz")

if st.button("Backend Test"):
    res = requests.get(f"{FASTAPI_URL}/")
    st.write(res.json())
