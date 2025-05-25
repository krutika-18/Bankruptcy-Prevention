import streamlit as st  
import joblib  
import numpy as np  
import plotly.graph_objects as go  
import matplotlib.pyplot as plt  
import pandas as pd  
import base64  

# ---- UI Enhancements ---- #
st.set_page_config(page_title="Bankruptcy Prevention", layout="wide")  
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #d4fc79, #96e6a1);
            color: black;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background: linear-gradient(to right, #d4fc79, #96e6a1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Load Model & Scaler ---- #
model = joblib.load('bankruptcy_model.pkl')  
scaler = joblib.load('scaler.pkl')  

# ---- Company Logo ---- #
st.sidebar.image("company_logo.png", width=200)


# ---- Sidebar Prevention Strategies ---- #
st.sidebar.title("💡 Prevention Strategies")  
st.sidebar.write("✔ Maintain a **healthy cash flow**")  
st.sidebar.write("✔ Reduce **unnecessary operational expenses**")  
st.sidebar.write("✔ Improve **financial flexibility**")  
st.sidebar.write("✔ Enhance **business credibility** to attract investors")  
st.sidebar.write("✔ Monitor **competitiveness** in the market")  

# ---- Light/Dark Mode Toggle ---- #
theme_mode = st.sidebar.radio("Select Theme:", ["🌞 Light Mode", "🌙 Dark Mode"])
if theme_mode == "🌙 Dark Mode":
    st.markdown(
        """
        <style>
            body { background: black; color: white; }
            .stApp { background: black; color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )

# ---- User Inputs ---- #
st.title("🛡️ Bankruptcy Prevention App")  
st.write("Analyze financial risk and get actionable strategies to prevent bankruptcy.")  

col1, col2 = st.columns(2)
with col1:
    industrial_risk = st.slider("Industrial Risk", 0.0, 1.0, 0.5)  
    management_risk = st.slider("Management Risk", 0.0, 1.0, 0.5)  
    financial_flexibility = st.slider("Financial Flexibility", 0.0, 1.0, 0.5)  

with col2:
    credibility = st.slider("Credibility", 0.0, 1.0, 0.5)  
    competitiveness = st.slider("Competitiveness", 0.0, 1.0, 0.5)  
    operating_risk = st.slider("Operating Risk", 0.0, 1.0, 0.5)  

# ---- Feature Processing ---- #
features_original = np.array([[industrial_risk, management_risk, financial_flexibility,  
                               credibility, competitiveness, operating_risk]])  
features_scaled = scaler.transform(features_original)  

total_risk = features_scaled[:, 0] + features_scaled[:, 1] + features_scaled[:, 5]
financial_stability = features_scaled[:, 2] + features_scaled[:, 3]

features_final = np.array([[*features_scaled[0], total_risk[0], financial_stability[0]]])

# ---- Prediction Output ---- #
if st.button("Check Bankruptcy Risk"):  
    prediction = model.predict(features_final)[0]  
    if prediction == 1:
        st.error("⚠️ High Bankruptcy Risk!")
        st.write("🔍 **Why?** Your financial risk factors indicate instability.")  
        st.write("💡 **Prevention Tip:** Focus on increasing financial flexibility and reducing operational risks.") 
        st.write("🏦 Seek investment or liquidity support.")

    else:
        st.success("✅ Low Bankruptcy Risk!")
        st.write("🎯 **Great!** Your company is financially stable.")  
        st.write("💡 **Tip:** Keep monitoring risk factors to maintain stability.")  

# ---- Live Risk Gauge (Speedometer) ---- #
st.subheader("📊 Live Bankruptcy Risk Gauge")
risk_score = model.predict_proba(features_final)[0][1] * 100  

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk_score,
    title={"text": "Bankruptcy Risk (%)"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "#ff4d4d"},
        "steps": [
            {"range": [0, 30], "color": "green"},
            {"range": [30, 70], "color": "yellow"},
            {"range": [70, 100], "color": "red"}
        ],
        "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": risk_score}
    }
))
st.plotly_chart(fig_gauge)

# ---- 3D Pie Chart (Risk Distribution) ---- #
st.subheader("📊 Bankruptcy Risk Distribution")  

labels = ["Bankruptcy Risk", "Non-Bankruptcy Risk"]
sizes = [risk_score, 100 - risk_score]  
colors = ["#ff4d4d", "#2ECC71"]  

fig_pie = go.Figure(data=[go.Pie(
    labels=labels, 
    values=sizes, 
    marker=dict(colors=colors),
    hole=0.3
)])
st.plotly_chart(fig_pie)

# ---- Bar Chart (Feature Importance) ---- #
st.subheader("📊 Feature Contribution to Bankruptcy Risk")  
feature_names = ["Industrial Risk", "Management Risk", "Financial Flexibility",  
                 "Credibility", "Competitiveness", "Operating Risk"]
feature_values = features_scaled[0] * 100  

fig_bar = go.Figure(data=[go.Bar(
    x=feature_names,  
    y=feature_values,  
    marker=dict(color=feature_values, colorscale="Bluered"),  
    text=[f"{val:.2f}%" for val in feature_values],  
    textposition="outside"
)])
st.plotly_chart(fig_bar)

import base64  # Ensure base64 is imported

# ---- Generate Downloadable Report ---- #
st.subheader("📄 Download Your Bankruptcy Risk Report")

def generate_report(risk_score, prediction, feature_values):
    report_text = f"""
    Bankruptcy Risk Analysis Report
    -----------------------------------
    Risk Score: {risk_score:.2f}%
    Prediction: {"High Risk" if prediction == 1 else "Low Risk"}
    
    Feature Contributions:
    - Industrial Risk: {feature_values[0]:.2f}
    - Management Risk: {feature_values[1]:.2f}
    - Financial Flexibility: {feature_values[2]:.2f}
    - Credibility: {feature_values[3]:.2f}
    - Competitiveness: {feature_values[4]:.2f}
    - Operating Risk: {feature_values[5]:.2f}
    """
    return report_text

# Ensure prediction and risk_score are calculated before calling the function
if st.button("Download Report", key="download_report"):
    prediction = model.predict(features_final)[0]  # Get the model prediction
    risk_score = model.predict_proba(features_final)[0][1] * 100  # Get risk probability
    feature_values = features_final[0]  # Extract feature values

    report = generate_report(risk_score, prediction, feature_values)  # ✅ Pass all required values
    b64 = base64.b64encode(report.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="bankruptcy_report.txt">📥 Click here to download</a>'
    st.markdown(href, unsafe_allow_html=True)