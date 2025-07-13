import streamlit as st
import joblib
import numpy as np

# Load saved model
model = joblib.load("titanic_model.pkl")

# Page title
st.title("Titanic Survival Prediction")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex (0 = Male, 1 = Female)", [0, 1])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation (0 = S, 1 = C, 2 = Q)", [0, 1, 2])

# Prediction
if st.button("Predict Survival"):
    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    prediction = model.predict(features)
    result = "Survived" if prediction[0] == 1 else "Did Not Survive"
    st.success(f"Prediction: {result}")