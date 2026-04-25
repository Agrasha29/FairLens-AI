import streamlit as st
import os
import google.generativeai as genai

# Safe way (works both local + cloud)
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

genai.configure(api_key=api_key)


def generate_fairness_explanation(fairness):

    model = genai.GenerativeModel("models/gemini-1.5-pro")

    prompt = f"""
    HR fairness analysis:

    Male Selection Rate: {fairness['Male Selection Rate']}
    Female Selection Rate: {fairness['Female Selection Rate']}
    Disparate Impact Ratio: {fairness['Disparate Impact Ratio']}

    Explain bias, risk, and recommendations clearly.
    """

    response = model.generate_content(prompt)
    return response.text