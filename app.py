import streamlit as st
import numpy as np
import joblib

# Step 2: Loading the trained model
model = joblib.load("diabetes_prediction.pkl")

# Step 3: Setting up Webpage
st.set_page_config(
  page_title="Diabetes Prediction", # Fixed typo in "Diabetes"
  page_icon='✨',
  layout='centered'
)

st.title("Diabetes Prediction App")
st.write("Enter Patient details here: ")
st.divider()

# Step 4: Input fields (Fixed default values to be within min and max)
pregnency = st.number_input('Pregnancies', 0, 20, 1)
glucose = st.number_input('Glucose', 0, 250, 100) 
bp = st.number_input('BloodPressure', 0, 225, 120) 
skin = st.number_input('SkinThickness', 0, 100, 20)
insulin = st.number_input('Insulin', 0, 900, 50) 
bmi = st.number_input('BMI', 0.0, 100.0, 25.0) # Changed to float
dpf = st.number_input('DiabetesPedigreeFunction', 0.000, 3.000, 0.500) # Changed to float
age = st.number_input('Age', 1, 120, 30)

# Prediction Button
if st.button('Predict Diabetes'):
  features = np.array([[
    pregnency,
    glucose,
    bp,
    skin,
    insulin,
    bmi,
    dpf,
    age
  ]])
  
  prediction = model.predict(features)[0]
  probability = model.predict_proba(features)[0]
  
  # Fixed Syntax Error Here
  if prediction == 1:
    st.error("High risk of Diabetes")
    st.write(f"Confidence Score: {probability[1]*100:.2f}%")
  else:
    st.success("No Risk")
    st.write(f"Confidence Score: {probability[0]*100:.2f}%") # Changed to probability[0] for 'No Risk' confidence
