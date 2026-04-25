import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def generate_fairness_explanation(fairness):

    prompt = f"""
    You are an HR fairness AI expert.

    Analyze this hiring system:

    Male Selection Rate: {fairness['Male Selection Rate']}
    Female Selection Rate: {fairness['Female Selection Rate']}
    Disparate Impact Ratio: {fairness['Disparate Impact Ratio']}

    Provide:
    1. Bias explanation
    2. Risk level
    3. Ethical impact
    4. Fix suggestions
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    return response.text