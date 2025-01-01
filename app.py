import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained models and scalers
svm_model = joblib.load('fish/svm_fish.pkl')
svm_scaler = joblib.load('fish/svm_scaler_fish.pkl')
perceptron_model = joblib.load('fish/perceptron_fish.pkl')
perceptron_scaler = joblib.load('fish/perceptron_scaler_fish.pkl')

# Application title
st.title("Iwak Prediction")

# Center the input form using custom CSS
st.markdown(
    """
    <style>
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Centered input container
st.markdown('<div class="centered">', unsafe_allow_html=True)

# Model selection
model_type = st.selectbox("Choose model", ["SVM", "Perceptron"])

# Input form
st.header("Input Features")
length = st.number_input("Fish Length (cm)", min_value=0.0, max_value=100.0, value=10.0, key='length')
weight = st.number_input("Fish Weight (g)", min_value=0.0, max_value=10000.0, value=200.0, key='weight')
w_l_ratio = weight / length if length > 0 else 0


# Prediction button
if st.button("Predict"):
    # Prepare the input data
    input_data = pd.DataFrame([[length, weight, w_l_ratio]], columns=['length', 'weight', 'w_l_ratio'])
    
    if model_type == "SVM":
        # Scale the input data for SVM model
        scaled_data = svm_scaler.transform(input_data)
        # Make prediction using SVM model
        prediction = svm_model.predict(scaled_data)
    else:
        # Scale the input data for Perceptron model
        scaled_data = perceptron_scaler.transform(input_data)
        # Make prediction using Perceptron model
        prediction = perceptron_model.predict(scaled_data)
    
    # Display prediction
    st.write("### Prediction Result")
    st.write(f"The predicted species is: **{prediction[0]}**")

st.markdown('</div>', unsafe_allow_html=True)
