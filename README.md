# FairLens-AI
AI system to detect, explain, and reduce bias in hiring using ML, SHAP, and Gemini AI.

# ⚖️ FairLens AI

### *AI-Powered Hiring Fairness Auditing & Bias Mitigation System*

---

## 🚀 Overview

**FairLens AI** is an intelligent fairness auditing system designed to detect, explain, and reduce bias in AI-driven hiring processes.

Modern hiring systems often inherit bias from historical data, leading to unfair decisions based on gender, age, or other sensitive attributes. FairLens AI solves this by making AI hiring systems **transparent, explainable, and fair**.

It goes beyond detection — it provides:

* Bias detection
* Explainable AI insights (SHAP)
* AI-generated fairness reports (Gemini AI)
* Bias mitigation simulation (before vs after impact)

---

## 🎯 Key Features

### ⚖️ 1. Fairness Detection

* Computes selection rates across demographic groups
* Calculates Disparate Impact Ratio
* Flags potential bias in hiring decisions

---

### 🧠 2. AI Explainability (SHAP)

* Feature importance analysis using SHAP
* Identifies bias-driving features (Gender, Age, Experience, etc.)
* Helps understand *why* the model makes decisions

---

### 🤖 3. AI Fairness Reports (Gemini AI)

* Converts technical metrics into HR-friendly insights
* Explains:

  * Who is affected
  * Ethical concerns
  * Risk analysis
  * Improvement recommendations

---

### 🛠 4. Bias Fix Engine

* Suggests actionable fairness improvements
* Simulates mitigation strategies
* Shows **before vs after fairness improvement**

---

### 📊 5. Interactive Dashboard (Streamlit)

* Fairness score gauge (0–100)
* Gender bias visualization charts
* Real-time model interpretation

---

### 📄 6. Automated Reporting

* Downloadable PDF fairness audit report
* CSV export of processed dataset

---

## 🏗️ System Architecture

Dataset → Preprocessing → ML Model Simulation
→ Fairness Metrics Engine
→ SHAP Explainability Layer
→ Gemini AI Reasoning Layer
→ Bias Fix Simulation Engine
→ Streamlit Dashboard

---

## 🧪 Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **ML Models:** Scikit-learn (Random Forest)
* **Explainability:** SHAP
* **AI Layer:** Google Gemini API
* **Visualization:** Matplotlib, Plotly
* **Data Processing:** Pandas, NumPy

---

## 📂 Project Structure

FairLens-AI/
│
├── app/
│   ├── frontend/
│   ├── backend/
│   ├── models/
│   ├── utils/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
├── tests/
├── main.py
├── requirements.txt
├── README.md

---

## ⚙️ Installation & Setup

### 1. Clone repository

```bash
git clone https://github.com/your-username/FairLens-AI.git
cd FairLens-AI
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add environment variables

Create `.env` file:

```
GEMINI_API_KEY=your_api_key_here
```

### 5. Run app

```bash
streamlit run app/frontend/streamlit_app.py
```

---

## 📊 Example Output

* Male Selection Rate: 0.31
* Female Selection Rate: 0.22
* Disparate Impact Ratio: 0.70

⚠️ Indicates potential gender bias in hiring decisions.

---

## 🧠 Impact

FairLens AI helps organizations:

* Detect hidden bias in AI systems
* Improve fairness before deployment
* Ensure ethical compliance
* Build trust in AI-driven hiring

---

## 🏆 Why This Project Matters

AI systems are increasingly used in hiring decisions — but without fairness checks, they can silently discriminate.

FairLens AI ensures:

> “AI decisions are not just accurate — but fair, explainable, and accountable.”

---

## 🔮 Future Improvements

* Multi-attribute fairness (race, age, disability)
* Real-time hiring API integration
* Automated bias correction pipeline
* Cloud deployment (GCP/AWS)
* Enterprise dashboard version

---

## 👨‍💻 Author

Agrasha Patel
B.Tech CSE (AIML) Student
AI/ML Enthusiast | Hackathon Builder

---

## ⭐ If you like this project

Give it a ⭐ on GitHub 🙌

