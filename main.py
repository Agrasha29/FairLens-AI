from app.utils.preprocessing import (
    load_dataset,
    basic_cleaning,
    detect_sensitive_attributes
)

from app.models.train_model import train_hiring_model


file_path = "data/raw/hiring_dataset.csv"

# Step 1: Load dataset
df = load_dataset(file_path)
print("\nDataset Loaded Successfully\n")
print(df.head())

# Step 2: Clean dataset
df = basic_cleaning(df)
# Create demo bias 
df.loc[
    (df["Gender"] == 0) & (df["ExperienceYears"] < 7),
    "HiringDecision"
] = 0
print("\nDataset Cleaned Successfully\n")

# Step 3: Detect sensitive columns
sensitive_columns = detect_sensitive_attributes(df)
print("\nDetected Sensitive Attributes:\n")
print(sensitive_columns)

# Step 4: Train ML Model
accuracy, model = train_hiring_model(df)
print("\nModel Training Successful\n")
print(f"Model Accuracy: {accuracy:.2f}")

# Step 5: Fairness Metrics
from app.models.fairness_metrics import calculate_gender_fairness

fairness_results = calculate_gender_fairness(df)
print("\nFairness Analysis Results\n")
for key, value in fairness_results.items():
    print(f"{key}: {value}")

# Step 6: SHAP Explainability
from app.models.explainability import generate_shap_explanation
shap_result = generate_shap_explanation(model, df)
print("\nExplainability Result\n")
print(shap_result)

# Step 7: Gemini Fairness Assistant
from app.backend.gemini_helper import generate_fairness_explanation
gemini_output = generate_fairness_explanation(fairness_results)
print("\nGemini Fairness Explanation\n")
print(gemini_output)