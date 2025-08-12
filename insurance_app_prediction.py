import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained models
with open("insurance_premium_model_rfr.pkl", "rb") as f:
    model = pickle.load(f)

with open("insurance_premium_model_lgbmr.pkl", "rb") as f1:
    model1 = pickle.load(f1)
    
with open("insurance_premium_model_gbr.pkl", "rb") as f2:
    model2 = pickle.load(f2)

# Define BMI calculation function
def calculate_bmi(height, weight):
    height_in_meters = height / 100  # Convert height from cm to meters
    return round(weight / (height_in_meters**2), 2)

st.title("Health Insurance Premium Prediction App")

# Age Categories - Fixed age range to match description (18-70)
age_bins = [18, 30, 40, 50, 60, 70]
age_labels = ['18-30', '30-40', '40-50', '50-60', '60+']
age_mapping = {'18-30': 0, '30-40': 1, '40-50': 2, '50-60': 3, '60+': 4}

with st.sidebar:
    st.header("User Information")
    # User inputs
    age = st.slider("Age", 17, 70, 35, help="Enter your age in years")
    st.caption("Age should be between 17 and 70.")
    age_cat = pd.cut([age], bins=age_bins, labels=age_labels, right=False)[0]
    age_category = age_mapping[age_cat]
    height = st.slider("Height", 150, 200, 170, help="Enter your height in cm")
    st.caption("Height should be between 150 and 200.")
    weight = st.slider("Weight", 50, 140, 70, help="Enter your weight in kg")
    st.caption("Weight should be between 50 and 140.")

with st.expander("Medical History"):
    options = {"No": 0, "Yes": 1}

    diabetes = st.selectbox("Diabetes", options.keys())
    blood_pressure_problems = st.selectbox("Blood Pressure Problems", options.keys())
    any_transplants = st.selectbox("Any Transplants", options.keys())
    any_chronic_diseases = st.selectbox("Any Chronic Diseases", options.keys())
    known_allergies = st.selectbox("Known Allergies", options.keys())
    history_of_cancer = st.selectbox("History of Cancer in Family", options.keys())
    number_of_surgeries = st.selectbox("Number of Major Surgeries", [0, 1, 2, 3])

    # Convert Yes/No to 0/1
    diabetes = options[diabetes]
    blood_pressure_problems = options[blood_pressure_problems]
    any_transplants = options[any_transplants]
    any_chronic_diseases = options[any_chronic_diseases]
    known_allergies = options[known_allergies]
    history_of_cancer = options[history_of_cancer]

# Calculate BMI and derived categories
bmi = calculate_bmi(height, weight)

# Streamlit color configuration
primaryColor = "#F39C12"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#333333"

# BMI Categories
bmi_bins = [0, 18.5, 24.9, 29.9, float('inf')]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
bmi_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}

bmi_category = pd.cut([bmi], bins=bmi_bins, labels=bmi_labels, right=False)[0]
bmi_code = bmi_mapping[bmi_category]

# Display calculated values
st.write(f"Selected Age: {age}")
st.write(f"Age Category: {age_cat} ({age_category})")
st.write(f"Calculated BMI: {bmi}")
st.write(f"BMI Category: {bmi_category} ({bmi_code})")

# Organize the input data - Fixed variable name
input_data = np.array([[age, diabetes, blood_pressure_problems, any_transplants, 
                       any_chronic_diseases, height, weight, known_allergies, 
                       history_of_cancer, number_of_surgeries, bmi, age_category, bmi_code]])

# Prediction
if st.button("Predict Insurance Premium"):
    prediction = model.predict(input_data)
    prediction1 = model1.predict(input_data)
    prediction2 = model2.predict(input_data)
    
    # Calculate ensemble average
    ensemble_prediction = (prediction[0] + prediction1[0] + prediction2[0]) / 3
    
    st.success(f"Predicted Premium Price: ₹{ensemble_prediction:.2f}")
    
    # Show individual model predictions for transparency
    with st.expander("Individual Model Predictions"):
        st.write(f"Random Forest: ₹{prediction[0]:.2f}")
        st.write(f"LightGBM: ₹{prediction1[0]:.2f}")
        st.write(f"Gradient Boosting: ₹{prediction2[0]:.2f}")
