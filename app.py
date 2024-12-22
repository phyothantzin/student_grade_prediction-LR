import streamlit as st
import numpy as np
import pickle

# Load the trained model and scaler
with open("student_model.pickle", "rb") as f:
    model = pickle.load(f)
with open("scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

# Define the UI
st.title("Student Final Grade Predictor")
st.write("Enter student details to predict their final grade (G3):")

# Input fields for user data
G1 = st.number_input("G1 (First Period Grade):", min_value=0, max_value=20, value=10)
G2 = st.number_input("G2 (Second Period Grade):", min_value=0, max_value=20, value=10)
studytime = st.selectbox("Study Time (Weekly):", options=[1, 2, 3, 4], format_func=lambda x: f"{x} hour(s)" if x == 1 else f"{x} hours")
failures = st.number_input("Number of Past Failures:", min_value=0, max_value=4, value=0)
freetime = st.slider("Free Time After School (1 - Very Low to 5 - Very High):", min_value=1, max_value=5, value=3)
goout = st.slider("Going Out with Friends (1 - Very Low to 5 - Very High):", min_value=1, max_value=5, value=3)
health = st.slider("Health Status (1 - Very Bad to 5 - Very Good):", min_value=1, max_value=5, value=3)
absences = st.number_input("Number of School Absences:", min_value=0, max_value=93, value=0)
Medu = st.selectbox("Mother's Education Level (0 - None to 4 - Higher):", options=[0, 1, 2, 3, 4])
Fedu = st.selectbox("Father's Education Level (0 - None to 4 - Higher):", options=[0, 1, 2, 3, 4])

# Additional Categorical Features
schoolsup = st.selectbox("Extra Educational Support (Yes/No):", options=["No", "Yes"])
famsup = st.selectbox("Family Educational Support (Yes/No):", options=["No", "Yes"])
paid = st.selectbox("Extra Paid Classes (Yes/No):", options=["No", "Yes"])
internet = st.selectbox("Internet Access at Home (Yes/No):", options=["No", "Yes"])
Dalc = st.slider("Workday Alcohol Consumption (1 - Very Low to 5 - Very High):", min_value=1, max_value=5, value=2)
Walc = st.slider("Weekend Alcohol Consumption (1 - Very Low to 5 - Very High):", min_value=1, max_value=5, value=2)

# Map categorical features to binary values
binary_mapping = {"No": 0, "Yes": 1}
schoolsup = binary_mapping[schoolsup]
famsup = binary_mapping[famsup]
paid = binary_mapping[paid]
internet = binary_mapping[internet]

# Create the input data array
input_data = np.array([[G1, G2, studytime, failures, freetime, goout, health, absences, Medu, Fedu, 
                        Dalc, Walc, schoolsup, famsup, paid, internet]])

# Apply scaling (use the same scaler used during training)
input_data = scaler.transform(input_data)

# Prediction button
if st.button("Predict Final Grade"):
    try:
        prediction = model.predict(input_data)
        prediction = np.clip(prediction, 0, None).round().astype(int)  # Ensure no negative grades
        st.success(f"Predicted Final Grade (G3): {prediction[0]}")
    except ValueError as e:
        st.error(f"Error: {str(e)}")
