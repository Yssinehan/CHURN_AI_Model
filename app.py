import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and preprocessor
@st.cache_resource
def load_model():
    model = joblib.load('rf_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_model()

st.title('🚀 Churn Prediction Dashboard')
st.markdown('Predict customer churn risk with Random Forest model (AUC 98%).')

# Feature inputs
st.header('Customer Features')

col1, col2 = st.columns(2)
with col1:
    st.subheader('Usage & Complaints')
    call_failure = st.number_input('Call Failure', min_value=0, max_value=50, value=7)
    complaints = st.number_input('Complaints', min_value=0, max_value=1, value=0)
    subscription_length = st.number_input('Subscription Length (months)', min_value=0, value=38)
    charge_amount = st.number_input('Charge Amount', min_value=0.0, value=100.0)
    seconds_use = st.number_input('Seconds of Use', min_value=0, value=500)
with col2:
    st.subheader('Frequency & Network')
    freq_use = st.number_input('Frequency of use', min_value=0, value=100)
    freq_sms = st.number_input('Frequency of SMS', min_value=0, value=50)
    distinct_calls = st.number_input('Distinct Called Numbers', min_value=0, value=5)
    age_group = st.selectbox('Age Group', options=[1,2,3,4,5], index=0)
    tariff_plan = st.selectbox('Tariff Plan', options=[1,2], index=0)
    status = st.selectbox('Status', options=[1,2], index=0)
    age = st.number_input('Age', min_value=15, max_value=55, value=30)
    customer_value = st.number_input('Customer Value', min_value=0.0, value=470.0)

if st.button('🔮 Predict Churn Risk'):
    # Create input df
    feature_names = ['Call Failure', 'Complaints', 'Subscription Length', 'Charge Amount',
                     'Seconds of Use', 'Frequency of use', 'Frequency of SMS',
                     'Distinct Called Numbers', 'Age Group', 'Tariff Plan', 'Status', 'Age', 'Customer Value']
    input_data = np.array([[call_failure, complaints, subscription_length, charge_amount,
                            seconds_use, freq_use, freq_sms, distinct_calls,
                            age_group, tariff_plan, status, age, customer_value]])
    input_df = pd.DataFrame(input_data, columns=feature_names)
    
    # Preprocess
    X_prep = preprocessor.transform(input_df)
    
    # Predict
    prob = model.predict_proba(X_prep)[0, 1]
    pred = model.predict(X_prep)[0]
    
    st.subheader('Prediction Results')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Churn Probability', f'{prob:.1%}')
    with col2:
        st.metric('Predicted Label', 'Churn' if pred == 1 else 'No Churn')
    with col3:
        risk = '🟢 Low' if prob < 0.3 else '🟡 Medium' if prob < 0.7 else '🔴 High'
        st.metric('Risk Level', risk)
    
    st.success('Dashboard ready for new customer predictions!')

