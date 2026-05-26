import os

class APIConfig:

    @staticmethod
    def get_groq_api_key():
        try:
            import streamlit as st
            return st.secrets["GROQ_API_KEY"]
        except:
            return os.getenv("GROQ_API_KEY")