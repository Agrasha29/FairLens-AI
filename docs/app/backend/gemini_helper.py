from google import genai
import os
from dotenv import load_dotenv

load_dotenv()


def generate_fairness_explanation(fairness_results):
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        prompt = f"""
        Explain hiring bias in simple HR terms:

        Male Selection Rate: {fairness_results['Male Selection Rate']}
        Female Selection Rate: {fairness_results['Female Selection Rate']}
        Disparate Impact Ratio: {fairness_results['Disparate Impact Ratio']}

        Include:
        1. Who is affected
        2. Why it is risky
        3. Ethical concerns
        4. Hiring recommendations
        5. How to improve fairness
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"AI explanation unavailable (fallback mode). Error: {str(e)}"