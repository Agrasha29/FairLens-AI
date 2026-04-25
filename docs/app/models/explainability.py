import shap
import matplotlib.pyplot as plt


def generate_shap_explanation(model, df):
    """
    Generate SHAP explanation for hiring model
    """

    target_column = "HiringDecision"

    X = df.drop(target_column, axis=1)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # Save SHAP summary plot
    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    plt.savefig("app/reports/shap_summary.png")
    plt.close()

    return "SHAP explanation generated successfully"