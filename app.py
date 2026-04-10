import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Student Marks Prediction App")

data = {
    'Hours': [1,2,3,4,5,6,7,8],
    'Marks': [10,20,30,40,50,60,70,80]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Marks']

model = LinearRegression()
model.fit(X, y)

hours = st.number_input("Enter study hours:", min_value=0.0, max_value=24.0, step=0.5)

if st.button("Predict"):
    prediction = model.predict([[hours]])
    st.success(f"Predicted Marks: {round(prediction[0], 2)}")