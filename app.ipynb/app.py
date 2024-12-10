#!/usr/bin/env python
# coding: utf-8





# In[3]:


import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('D:/Capstone2/models/final_model.pkl')

# App title
st.title("Credit Card Fraud Detection")

# Input form
st.write("Enter transaction details:")
scaled_time = st.number_input("Scaled Time")
scaled_amount = st.number_input("Scaled Amount")
V1=st.number_input("V1")
V2=st.number_input("V2")
V3=st.number_input("V3")
V4=st.number_input("V4")
V5=st.number_input("V5")
V6=st.number_input("V6")
V7=st.number_input("V7")
V8=st.number_input("V8")
V9=st.number_input("V9")
V10=st.number_input("V10")
V11=st.number_input("V11")
V12=st.number_input("V12")
V13=st.number_input("V13")
V14=st.number_input("V14")
V16=st.number_input("V!^")
V17=st.number_input("V17")
V18=st.number_input("V18")
V19=st.number_input("V19")
V20=st.number_input("V20")
V21=st.number_input("V21")
V23=st.number_input("V22")
V24=st.number_input("V23")
V25=st.number_input("V25")
V26=st.number_input("V26")
V27=st.number_input("V27")
V28=st.number_input("v28")
# Add inputs for other features

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "scaled_time": scaled_time,
        "scaled_amount": scaled_amount,
        # Add other feature values
    }])
    prediction = model.predict(input_data)[0]
    result = "Fraudulent" if prediction == 1 else "Non-Fraudulent"
    st.write(f"Prediction: {result}")


# In[ ]:





# In[ ]:





# In[ ]:




