import shap
import numpy as np

def get_shap_explanation(model, X):
    """
    Returns SHAP values for feature importance
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return shap_values