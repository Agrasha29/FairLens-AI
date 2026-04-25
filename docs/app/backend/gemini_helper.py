import os
import google.generativeai as genai
import streamlit as st

api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

genai.configure(api_key=api_key)


def generate_fairness_explanation(fairness):

    model = genai.GenerativeModel("models/gemini-1.5-flash")

    prompt = f"""
    HR fairness analysis:

    Male Selection Rate: {fairness['Male Selection Rate']}
    Female Selection Rate: {fairness['Female Selection Rate']}
    Disparate Impact Ratio: {fairness['Disparate Impact Ratio']}
    """

    response = model.generate_content(prompt)
    return response.text