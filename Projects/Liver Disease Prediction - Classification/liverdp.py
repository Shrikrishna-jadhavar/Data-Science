# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 23:13:58 2024

@author: Shrikrishna Jadhavar
"""

import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load('liver_disease_model.pkl')

# Function to get user inputs
def get_user_input():
    age = st.number_input('Age', min_value=1, max_value=100, value=30)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    albumin = st.number_input('Albumin', min_value=0.0, max_value=100.0, value=4.0)
    alkaline_phosphatase = st.number_input('Alkaline Phosphatase', min_value=0.0, max_value=1000.0, value=30.0)
    alanine_aminotransferase = st.number_input('Alanine Aminotransferase', min_value=0.0, max_value=1000.0, value=30.0)
    aspartate_aminotransferase = st.number_input('Aspartate Aminotransferase', min_value=0.0, max_value=1000.0, value=30.0)
    bilirubin = st.number_input('Total Bilirubin', min_value=0.0, max_value=100.0, value=1.0)
    cholinesterase = st.number_input('Cholinesterase', min_value=0.0, max_value=100.0, value=30.0)
    cholesterol = st.number_input('Cholesterol', min_value=0.0, max_value=100.0, value=30.0)
    creatinina = st.number_input('Creatinina', min_value=0.0, max_value=1000.0, value=1.0)
    gamma_glutamyl_transferase = st.number_input('Gamma Glutamyltransferase', min_value=0.0, max_value=1000.0, value=30.0)
    protein = st.number_input('Protein', min_value=0.0, max_value=100.0, value=30.0)

    # Collect inputs into a dictionary
    user_data = user_data = {   
        'age': age,
        'sex': sex,
        'albumin': albumin,
        'alkaline_phosphatase': alkaline_phosphatase,
        'alanine_aminotransferase': alanine_aminotransferase,
        'aspartate_aminotransferase': aspartate_aminotransferase,
        'bilirubin': bilirubin,
        'cholinesterase': cholinesterase,
        'cholesterol': cholesterol,
        'creatinina': creatinina,
        'gamma_glutamyl_transferase': gamma_glutamyl_transferase,
        'protein': protein
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

# Preprocessing function
def preprocess_data(input_data):
    # One-hot encode 'gender': Create two columns: 'gender_Male' and 'gender_Female'
    input_data['sex'] = input_data['sex'].map({'Male': 1, 'Female': 0})
    
    #input_data = pd.get_dummies(input_data, columns=['sex'], drop_first=False)
    
    # Ensure all columns are present even if one category is missing in the input
    #expected_columns = ['age', 'sex_Male', 'sex_Female', 'albumin',  'alkaline_phosphatase', 'alanine_aminotransferase'
     #                   'aspartate_aminotransferase', 'bilirubin','cholinesterase', 'cholesterol', 'creatinina',
      #                  'gamma_glutamyl_transferase', 'protein']
    #for col in expected_columns:
     #   if col not in input_data.columns:
      #      input_data[col] = 0  # Set missing columns to 0
    
    return input_data

# Streamlit app logic
st.title('Liver Disease Prediction')

# Get user input
user_input = get_user_input()

# Preprocess the input data (convert categorical to numeric)
processed_data = preprocess_data(user_input)

# Check the number of features
st.write(f"Processed data shape: {processed_data.shape}")

# Make predictions if the user clicks the 'Predict' button
if st.button('Predict'):
    # Ensure that processed_data is defined and contains the expected features
    prediction = model.predict(processed_data)
    
    # Display the result
    if prediction[0] == 1:
        st.write("The patient is likely to have liver disease.")
    else:
        st.write("The patient is unlikely to have liver disease.")
