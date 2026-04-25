
import streamlit as st
import joblib

model = joblib.load("model.pkl")

st.title("Placement Predictor")

cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
skills = st.slider("Skills", 0, 5, 2)
internships = st.slider("Internships", 0, 3, 1)
projects = st.slider("Projects", 0, 6, 2)

if st.button("Predict"):
    pred = model.predict([[cgpa, skills, internships, projects]])
    st.write("Eligible" if pred[0]==1 else "Not Eligible")
