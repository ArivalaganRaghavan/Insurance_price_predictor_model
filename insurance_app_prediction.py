import streamlit as st
import pickle
import numpy as np
import pandas as pd

#load the trained model
with open("insurance_premium_model_rfr.pkl", "rb") as f:
    model = pickle.load(f)

###with open("insurance_premium_model_lgbmr.pkl", "rb") as f1:
    ### model1 = pickle.load(f1)

with open("insurance_premium_model_gbr.pkl", "rb") as f2:
    model2 = pickle.load(f2)

# define BMI caclulation
def calculate_bmi(height, weight):
    height_in_meters = height /100 # Convert height from cm to meters
    return round(weight / (height_in_meters**2),2)

st.title("health Insurance Premium prediction app")

# Age Categories
age_bins = [8, 30, 40,50,60,70]
age_labels = ['18-30', '30-40', '40-50', '50-60', '60+']
age_mapping = {'18-30': 0, '30-40': 1, '40-50': 2, '50-60': 3, '60+': 4}


## usre inputs
age = st.slider("age", 17,70,35)
age_cat = pd.cut([age], bins=age_bins, labels=age_labels, right=False)[0]
age_category = age_mapping[age_cat]

diabetes = st.selectbox("Diabetes", [0,1])
blood_pressure_problems = st.selectbox("BloodPressureProblems", [0,1])
any_transplants = st.selectbox("AnyTransplants",[0,1])
any_chronic_diseases = st.selectbox("AnyChronicDiseases", [0,1])
height = st.slider("height", 150,200,170)
weight = st.slider("weight", 50,120,70)
known_allergies  = st.selectbox("KnownAllergies", [0,1])
history_of_cancer = st.selectbox("HistoryOfCancerInFamily", [0,1])
number_of_surgeries  = st.selectbox("NumberOfMajorSurgeries", [0,1,2,3])

# Calculate BMI and derived categories
bmi = calculate_bmi(height, weight)


# BMI Categories
bmi_bins = [0, 18.5, 24.9, 29.9, float('inf')]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
bmi_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}

bmi_category = pd.cut([bmi], bins=bmi_bins, labels=bmi_labels, right=False)[0] 
bmi_code = bmi_mapping[bmi_category]

bmi_mapping[bmi_category] 
st.write(f"Selected Age: {age}") 
st.write(f"Age Category: {age_cat} ({age_category})") 
st.write(f"Selected BMI: {bmi}")
st.write(f"BMI Category: {bmi_category} ({bmi_code})")



# organize the input data
input_date = np.array([[age, diabetes, blood_pressure_problems, any_transplants, any_chronic_diseases, height, weight, known_allergies, history_of_cancer, number_of_surgeries,bmi, age_category, bmi_code]])

# Prediction

if st.button("Predict Insurance Premium"):
    prediction = model.predict(input_date)
    prediction2 = model2.predict(input_date)
    st.write(f"Random Forest Regressor Predicted Premium Price: ${prediction[0]:.2f}")
    st.write(f"Gradient Boosting Regressor Model Predicted Premium Price: ${prediction2[0]:.2f}")
    