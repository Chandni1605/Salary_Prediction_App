#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:29:08 2024

@author: chandnisingh
"""

import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("/Users/chandnisingh/Desktop/spyder projects/linear_regression_model.pkl","rb"))

st.title("Salary Prediction App")

st.write("This app predicts the salary based on years of experience using simple linear regression model")

years_experience = st.number_input("Enter Years of experience:" , min_value = 0.0, max_value = 50.0, value=1.0,step=0.5)

if st.button("Predict Salary"):
    experience_input = np.array([[years_experience]])  # Convert the input to a 2D array for prediction
    prediction = model.predict(experience_input)
   
    # Display the result
    st.success(f"The predicted salary for {years_experience} years of experience is: ${prediction[0]:,.2f}")
   
# Display information about the model
st.write("The model was trained using a dataset of salaries and years of experience.")