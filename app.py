import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Model
model = LinearRegression()
model.fit(X, y)

# UI
st.title("Student Marks Prediction App")

hours = st.number_input("Enter study hours:")

if st.button("Predict"):
    prediction = model.predict(np.array([[hours]]))
    st.success(f"Predicted Marks: {prediction[0]:.2f}")
