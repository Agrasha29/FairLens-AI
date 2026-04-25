def generate_bias_fix_suggestions(fairness):

    male = fairness["Male Selection Rate"]
    female = fairness["Female Selection Rate"]
    di = fairness["Disparate Impact Ratio"]

    suggestions = []

    # Case 1: strong gender imbalance
    if di < 0.8:
        suggestions.append("⚠️ Severe bias detected against female candidates.")

        suggestions.append("✔ Remove or anonymize Gender feature during training.")
        suggestions.append("✔ Apply re-sampling (SMOTE or balanced dataset).")
        suggestions.append("✔ Use fairness-aware algorithms (Fairlearn / AIF360).")
        suggestions.append("✔ Adjust decision threshold for fairness parity.")

    # Case 2: mild imbalance
    elif di < 0.95:
        suggestions.append("⚠️ Mild bias detected in hiring outcomes.")

        suggestions.append("✔ Monitor selection rates per gender regularly.")
        suggestions.append("✔ Apply light reweighting of training samples.")
        suggestions.append("✔ Review feature importance for bias leakage.")

    # Case 3: fair system
    else:
        suggestions.append("✅ System is reasonably fair.")
        suggestions.append("✔ Continue monitoring fairness metrics over time.")
        suggestions.append("✔ Ensure periodic bias audits.")

    return suggestions