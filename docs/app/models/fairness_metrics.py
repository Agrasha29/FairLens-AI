import pandas as pd


def calculate_gender_fairness(df):
    """
    Calculate fairness using Gender column
    Assumption:
    Gender = 1 → Male
    Gender = 0 → Female

    HiringDecision:
    1 = Selected
    0 = Rejected
    """

    # Male selection rate
    male_selected = df[df["Gender"] == 1]["HiringDecision"].mean()

    # Female selection rate
    female_selected = df[df["Gender"] == 0]["HiringDecision"].mean()

    # Disparate Impact Ratio
    # (Protected group selection rate / Privileged group selection rate)
    if male_selected != 0:
        disparate_impact = female_selected / male_selected
    else:
        disparate_impact = 0

    results = {
        "Male Selection Rate": round(male_selected, 2),
        "Female Selection Rate": round(female_selected, 2),
        "Disparate Impact Ratio": round(disparate_impact, 2)
    }

    return results