import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# Load
model = tf.keras.models.load_model("churn_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("Customer Churn Prediction")

# Inputs
age = st.number_input("Age", 18, 100, 30)
tenure = st.number_input("Tenure", 0, 100, 10)
monthlycharges = st.number_input("Monthly Charges", 0.0, 100000.0, 5000.0)
totalcharges = st.number_input("Total Charges", 0.0, 1000000.0, 20000.0)
credit_score = st.number_input("Credit Score", 300, 900, 600)

if st.button("Predict"):

    input_dict = {col: 0 for col in columns}

    input_dict['age'] = age
    input_dict['tenure'] = tenure
    input_dict['monthlycharges'] = monthlycharges
    input_dict['totalcharges'] = totalcharges
    input_dict['credit_score'] = credit_score

    input_data = np.array([list(input_dict.values())])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0][0]

    if prediction > 0.5:
        st.write("Customer will CHURN")
    else:
        st.write("Customer will STAY")