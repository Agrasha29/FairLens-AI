from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_fairness_explanation(fairness):

    prompt = f"""
    HR fairness analysis:

    Male Selection Rate: {fairness['Male Selection Rate']}
    Female Selection Rate: {fairness['Female Selection Rate']}
    Disparate Impact Ratio: {fairness['Disparate Impact Ratio']}

    Explain:
    - bias detection
    - risk level
    - ethical concerns
    - recommendations
    """

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",   # ✅ FIXED
        contents=prompt
    )

    return response.text