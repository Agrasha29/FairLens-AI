import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import shap
import plotly.graph_objects as go

# ---------------- FIX PATH ----------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ---------------- IMPORTS ----------------
from app.utils.preprocessing import basic_cleaning
from app.models.fairness_metrics import calculate_gender_fairness
from app.backend.gemini_helper import generate_fairness_explanation
from app.backend.bias_fix_ai import generate_bias_fix_suggestions
from app.utils.pdf_report import generate_pdf_report


# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="FairLens AI", layout="wide")

st.title("⚖️ FairLens AI - Bias Detection System")

uploaded_file = st.file_uploader("Upload Hiring Dataset", type=["csv"])


# ---------------- SESSION STATE ----------------
if "gemini_text" not in st.session_state:
    st.session_state.gemini_text = None


# =========================================================
# MAIN APP
# =========================================================
if uploaded_file:

    # ---------------- LOAD DATA ----------------
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Dataset")
    st.dataframe(df.head())

    # ---------------- CLEAN DATA ----------------
    df = basic_cleaning(df)
    df = df.dropna()
    st.success("Dataset Cleaned Successfully")

    # ---------------- MODEL SIMULATION ----------------
    if "HiringDecision" not in df.columns:
        st.info("Generating Hiring Decision using rule-based logic...")
        df["HiringDecision"] = (df["SkillScore"] > 50).astype(int)

    # ---------------- FAIRNESS ANALYSIS ----------------
    fairness = calculate_gender_fairness(df)

    st.subheader("⚖️ Fairness Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Male Selection Rate", f"{fairness['Male Selection Rate']:.2f}")
    col2.metric("Female Selection Rate", f"{fairness['Female Selection Rate']:.2f}")
    col3.metric("Disparate Impact", f"{fairness['Disparate Impact Ratio']:.2f}")

    di = fairness["Disparate Impact Ratio"]

    # ---------------- FAIRNESS SCORE ----------------
    fairness_score = max(0, min(di, 1)) * 100
    fairness_score = round(fairness_score, 2)

    st.subheader("🧠 Fairness Score Dashboard")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=fairness_score,
        title={'text': "Fairness Score"},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#4C78A8"},
            'steps': [
                {'range': [0, 50], 'color': "#F8696B"},
                {'range': [50, 80], 'color': "#FFEB84"},
                {'range': [80, 100], 'color': "#63BE7B"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # 🛠 BIAS FIX AI
    # =====================================================
    st.subheader("🛠 AI Bias Fix Recommendations")

    if st.button("Generate Fix Suggestions"):
        fixes = generate_bias_fix_suggestions(fairness)
        for fix in fixes:
            st.write("•", fix)

    # =====================================================
    # 📊 SHAP EXPLAINABILITY
    # =====================================================
    st.subheader("📊 AI Explainability (SHAP Feature Impact)")

    if st.checkbox("Show Why Model Makes Decisions"):

        try:
            X = df.drop("HiringDecision", axis=1)

            # keep only numeric features (IMPORTANT FIX)
            X = X.select_dtypes(include=["int64", "float64"])
            y = df["HiringDecision"]

            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            st.write("🔍 Feature importance driving hiring decisions:")

            fig = plt.figure()

            shap.plots.bar(
                shap.Explanation(
                    values=shap_values[1],
                    base_values=explainer.expected_value[1],
                    data=X,
                    feature_names=X.columns
                ),
                show=False
            )

            st.pyplot(fig)

        except Exception as e:
            st.error(f"SHAP failed: {e}")

    # =====================================================
    # 📊 VISUALIZATION
    # =====================================================
    st.subheader("📊 Gender Bias Visualization")

    labels = ["Male", "Female"]
    values = [
        fairness["Male Selection Rate"],
        fairness["Female Selection Rate"]
    ]

    fig, ax = plt.subplots()

    bars = ax.bar(labels, values)

    ax.set_ylabel("Selection Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Hiring Bias Comparison")

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f"{height:.2f}", ha="center")

    st.pyplot(fig)

    # =====================================================
    # 🚨 BIAS STATUS
    # =====================================================
    st.subheader("🚨 Bias Detection Status")

    if di < 0.8:
        st.error("Bias Detected in Hiring System")
    else:
        st.success("No Major Bias Detected")

    # =====================================================
    # 🤖 GEMINI AI REPORT (SESSION STATE FIXED)
    # =====================================================
    if st.button("🤖 Generate AI Fairness Report"):

        with st.spinner("Analyzing with Gemini AI..."):
            st.session_state.gemini_text = generate_fairness_explanation(fairness)

        st.subheader("🧠 Gemini AI Explanation")
        st.write(st.session_state.gemini_text)

    # =====================================================
    # 📄 PDF REPORT
    # =====================================================
    if st.button("📄 Download AI Bias Report (PDF)"):

        if st.session_state.gemini_text is None:
            st.session_state.gemini_text = generate_fairness_explanation(fairness)

        file_path = "fairlens_report.pdf"

        generate_pdf_report(
            file_path,
            fairness,
            fairness_score,
            st.session_state.gemini_text
        )

        with open(file_path, "rb") as f:
            st.download_button(
                label="⬇ Download PDF Report",
                data=f,
                file_name="FairLens_AI_Report.pdf",
                mime="application/pdf"
            )

    # =====================================================
    # 📦 CSV DOWNLOAD
    # =====================================================
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Processed Dataset",
        data=csv,
        file_name="fairlens_report.csv",
        mime="text/csv"
    )