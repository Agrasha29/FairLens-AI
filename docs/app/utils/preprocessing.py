import pandas as pd


def load_dataset(file_path):
    """
    Load dataset from CSV file
    """
    df = pd.read_csv(file_path)
    return df


def basic_cleaning(df):
    """
    Basic cleaning:
    - remove duplicates
    - handle missing values
    """
    df = df.drop_duplicates()

    # fill missing values with mode for simplicity
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            df[column].fillna(df[column].mode()[0], inplace=True)

    return df


def detect_sensitive_attributes(df):
    """
    Detect sensitive columns automatically
    """
    sensitive_keywords = [
        "gender",
        "age",
        "caste",
        "religion",
        "marital",
        "disability",
        "location"
    ]

    detected = []

    for col in df.columns:
        col_lower = col.lower()

        for keyword in sensitive_keywords:
            if keyword in col_lower:
                detected.append(col)

    return detected