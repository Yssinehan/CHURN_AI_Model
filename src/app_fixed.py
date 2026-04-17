import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pdfplumber
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from report_generator import generate_pdf_report

# Page config
st.set_page_config(
    page_title="Churn Risk Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        min-height: 100vh;
        padding: 2rem;
    }
    
    .stApp {
        background: transparent !important;
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #ffffff;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .card {
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    .card:hover {
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .metric-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        border-left: 4px solid #3b82f6;
        font-family: 'Inter', sans-serif;
    }
    
    .btn-primary {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        font-size: 16px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    .btn-primary:hover {
        background: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    .risk-low { border-left-color: #10b981; }
    .risk-medium { border-left-color: #f59e0b; }
    .risk-high { border-left-color: #ef4444; }
    
    .slider-container {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .section-header {
        color: #1e3a8a;
        font-weight: 600;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('model/rf_model.pkl')
    preprocessor = joblib.load('model/preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_model()

# State for live prediction
if 'input_data' not in st.session_state:
    st.session_state.input_data = np.zeros((1, 13))

# Helper function to parse PDF text
def parse_pdf_features(text):
    """Parse PDF to extract 13 client features."""

    patterns = {
        'Call Failure': r'Call Failure[: ]*([0-9.]+)',
        'Complaints': r'Complaints[: ]*([0-9.]+)',
        'Subscription Length': r'Subscription Length[: ]*([0-9.]+)',
        'Charge Amount': r'Charge Amount[: ]*([0-9.]+)',
        'Seconds of Use': r'Seconds of Use[: ]*([0-9.]+)',
        'Frequency of use': r'Frequency of use[: ]*([0-9.]+)',
        'Frequency of SMS': r'Frequency of SMS[: ]*([0-9.]+)',
        'Distinct Called Numbers': r'Distinct Called Numbers[: ]*([0-9.]+)',
        'Age Group': r'Age Group[: ]*([1-5])',
        'Tariff Plan': r'Tariff Plan[: ]*([1-2])',
        'Status': r'Status[: ]*([1-2])',
        'Age': r'Age[: ]*([0-9.]+)',
        'Customer Value': r'Customer Value[: ]*([0-9.]+)',
    }
    feature_data = {}
    import re
    for feature, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            feature_data[feature] = float(match.group(1))
        else:
            raise ValueError(f"Feature '{feature}' not found in PDF")
    return feature_data

# Sidebar inputs
st.sidebar.markdown('<div class="card"><h3>Customer Profile</h3></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="card p-4">', unsafe_allow_html=True)
    
    st.markdown('<h4 class="section-header mb-3">Usage Metrics</h4>', unsafe_allow_html=True)
    usage_col1, usage_col2 = st.columns(2)
    with usage_col1:
        call_failure = st.slider('Call Failure', 0, 50, 7, key='cf')
        complaints = st.slider('Complaints', 0, 1, 0, key='comp')
    with usage_col2:
        subscription_length = st.slider('Subscription Length (months)', 0, 60, 38, key='sub')
        seconds_use = st.slider('Seconds of Use', 0, 2000, 500, key='sec')
    
    st.markdown('---')
    
    freq_col1, freq_col2 = st.columns(2)
    with freq_col1:
        freq_use = st.slider('Frequency of Use', 0, 500, 100, key='fu')
        freq_sms = st.slider('SMS Frequency', 0, 300, 50, key='sms')
    with freq_col2:
        distinct_calls = st.slider('Distinct Called Numbers', 0, 20, 5, key='dc')
        customer_value = st.slider('Customer Value', 0.0, 1000.0, 470.0, key='cv')
    
    st.markdown('---')
    
    dem_col1, dem_col2 = st.columns(2)
    with dem_col1:
        age_group = st.selectbox('Age Group', [1,2,3,4,5], 0, key='ag')
        tariff_plan = st.selectbox('Tariff Plan', [1,2], 0, key='tp')
        status = st.selectbox('Status', [1,2], 0, key='stat')
    with dem_col2:
        age = st.slider('Age', 15, 55, 30, key='age')
        charge_amount = st.slider('Charge Amount', 0.0, 500.0, 100.0, key='ca')
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # PDF Upload Feature
    uploaded_file = st.file_uploader("Choisir un fichier PDF rapport client", type=['pdf'])
    if uploaded_file is not None:
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            features = parse_pdf_features(text)
            # Auto-fill sliders
            st.session_state['cf'] = int(features.get('Call Failure', 7))
            st.session_state['comp'] = int(features.get('Complaints', 0))
            st.session_state['sub'] = int(features.get('Subscription Length', 38))
            st.session_state['ca'] = float(features.get('Charge Amount', 100))
            st.session_state['sec'] = int(features.get('Seconds of Use', 500))
            st.session_state['fu'] = int(features.get('Frequency of use', 100))
            st.session_state['sms'] = int(features.get('Frequency of SMS', 50))
            st.session_state['dc'] = int(features.get('Distinct Called Numbers', 5))
            st.session_state['ag'] = int(features.get('Age Group', 1))
            st.session_state['tp'] = int(features.get('Tariff Plan', 1))
            st.session_state['stat'] = int(features.get('Status', 1))
            st.session_state['age'] = int(features.get('Age', 30))
            st.session_state['cv'] = float(features.get('Customer Value', 470))
            st.success("📄 PDF parsed! Sliders auto-filled.")
        except Exception as e:
            st.error(f"PDF parsing error: {str(e)}")

# Update input data live
def update_input():
    feature_names = ['Call Failure', 'Complaints', 'Subscription Length', 'Charge Amount',
                     'Seconds of Use', 'Frequency of use', 'Frequency of SMS',
                     'Distinct Called Numbers', 'Age Group', 'Tariff Plan', 'Status', 'Age', 'Customer Value']
    input_data = np.array([[ 
        st.session_state['cf'], st.session_state['comp'], st.session_state['sub'], st.session_state['ca'],
        st.session_state['sec'], st.session_state['fu'], st.session_state['sms'],
        st.session_state['dc'], st.session_state['ag'], st.session_state['tp'], st.session_state['stat'], 
        st.session_state['age'], st.session_state['cv']
    ]])
    st.session_state.input_data = input_data

# Main content
st.markdown('<div class="card"><h1>Churn Risk Assessment Dashboard</h1><p>ML-powered customer churn prediction and risk analysis</p></div>', unsafe_allow_html=True)

# Live prediction button
col_btn1, col_btn2 = st.columns([3,1])
with col_btn1:
    st.markdown('<h2 style="color: white;">Live Prediction</h2>', unsafe_allow_html=True)
with col_btn2:
    if st.button('Predict Churn Risk', key='predict', help='Change sliders for live updates', use_container_width=True):
        update_input()

# Prediction
input_df = pd.DataFrame(st.session_state.input_data, columns=['Call Failure', 'Complaints', 'Subscription Length', 'Charge Amount',
                                                              'Seconds of Use', 'Frequency of use', 'Frequency of SMS',
                                                              'Distinct Called Numbers', 'Age Group', 'Tariff Plan', 'Status', 'Age', 'Customer Value'])

# Fix dtypes for categorical columns to match training
input_df['Age Group'] = input_df['Age Group'].astype(int)
input_df['Tariff Plan'] = input_df['Tariff Plan'].astype(int)
input_df['Status'] = input_df['Status'].astype(int)

try:
    X_prep = preprocessor.transform(input_df)
except ValueError as e:
    st.error(f"Prediction error: {e}. Try changing categorical sliders.")
    st.stop()

prob = model.predict_proba(X_prep)[0, 1]
pred = model.predict(X_prep)[0]

# Results row
col1, col2, col3 = st.columns(3)
with col1:
    risk_class = 'risk-low' if prob < 0.3 else 'risk-medium' if prob < 0.7 else 'risk-high'
    st.markdown(f'''
    <div class="metric-card {risk_class}">
        <h3 style="color: #1e3a8a; font-size: 1.1rem;">Churn Probability</h3>
        <h1 style="color: #1e40af; font-size: 2.5rem;">{prob:.1%}</h1>
    </div>
    ''', unsafe_allow_html=True)
with col2:
    label = 'No Churn' if pred == 0 else 'Churn'
    label_color = '#10b981' if pred == 0 else '#ef4444'
    st.markdown(f'''
    <div class="metric-card">
        <h3 style="color: #1e3a8a; font-size: 1.1rem;">Prediction</h3>
        <h2 style="color: {label_color}; font-size: 2rem;">{label}</h2>
    </div>
    ''', unsafe_allow_html=True)
with col3:
    risk_level = 'Low Risk' if prob < 0.3 else 'Medium Risk' if prob < 0.7 else 'High Risk'
    st.markdown(f'''
    <div class="metric-card {risk_class}">
        <h3 style="color: #1e3a8a; font-size: 1.1rem;">Risk Level</h3>
        <h2 style="color: #1e40af; font-size: 2rem;">{risk_level}</h2>
    </div>
    ''', unsafe_allow_html=True)

# Interactive Plotly Gauge
fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = prob,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Churn Risk Gauge"},
    delta = {'reference': 0.5},
    gauge = {
        'axis': {'range': [None, 1]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 0.3], 'color': "lightgreen"},
            {'range': [0.3, 0.7], 'color': "yellow"},
            {'range': [0.7, 1], 'color': "red"}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 0.7}}))

fig_gauge.update_layout(height=300, font=dict(color="white"))
st.plotly_chart(fig_gauge, use_container_width=True)

# Feature importance - dynamic length
importances = model.feature_importances_
feature_names = [
    'Call Failure', 'Complaints', 'Subscription Length', 'Charge Amount',
    'Seconds of Use', 'Frequency of use', 'Frequency of SMS',
    'Distinct Called Numbers', 'Age Group', 'Tariff Plan', 'Status', 'Age', 'Customer Value'
]
min_len = min(len(importances), len(feature_names))
fi_df = pd.DataFrame({
    'feature': feature_names[:min_len],
    'importance': importances[:min_len]
}).sort_values('importance', ascending=True)

fig_bar = px.bar(
    fi_df, 
    x='importance', 
    y='feature', 
    orientation='h',
    title="Model Feature Importance",
    color='importance',
    color_continuous_scale=['#1e3a8a', '#3b82f6', '#60a5fa']
)
fig_bar.update_layout(height=400, font=dict(color="white"), title_font_color="white")
st.plotly_chart(fig_bar, use_container_width=True)

# Expander for details
with st.expander("Détails du Modèle", expanded=False):
    st.markdown("""
    **Comment ça marche ?**
    - Le modèle analyse 13 informations sur le client (appels, plaintes, durée abonnement, âge, valeur...)
    - Il prédit la probabilité de départ (churn) en % avec précision de 98%
    - Vert = faible risque, Jaune = moyen, Rouge = haut risque
    - Changez les curseurs pour voir les prédictions en temps réel
    
    **Actions recommandées :**
    - Risque faible : Maintenir relation normale
    - Risque moyen : Contact personnalisé
    - Risque haut : Offre spéciale immédiate
    """)

