# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained model 
model_data = joblib.load("intern1_it_jobs_model.joblib")
model = model_data["model"]
label_encoder = model_data["label_encoder"]
columns = model_data["columns"]

# Streamlit app title
st.title("🎓 IT Job Role Recommender (Beginner Demo)")
st.write("This app recommends a suitable IT job role for you based on your background, skills, and interests. 🚀")

# User Inputs 
age = st.slider("Age", 18, 50, 25)
education = st.selectbox("Education level", ["High School", "Diploma", "B.Tech", "M.Tech", "PhD", "Other"])
background = st.selectbox("Background", ["CS", "IT", "ECE", "Mechanical", "Commerce", "Other"])
prog = st.selectbox("Programming Language", ["Python", "Java", "C++", "JavaScript", "None"])
tool = st.selectbox("Tool Known", ["Power BI", "Excel", "Tableau", "TensorFlow", "PyTorch", "None"])
interest = st.selectbox("Interest Area", ["AI", "Web Development", "Cybersecurity", "Data Analysis", "Mobile", "DevOps"])
study_hours = st.slider("Study Hours per Day", 0, 12, 3)
preferred_time = st.selectbox("Preferred Study Time", ["Morning", "Night", "Anytime"])
learning_style = st.selectbox("Learning Style", ["Visual", "Hands-on", "Reading", "Mixed"])
soft_skill = st.selectbox("Soft Skill", ["Problem Solving", "Communication", "Leadership", "Teamwork", "Time Management"])
preferred_domain = st.selectbox("Preferred Domain", ["IT", "Govt Job", "Startup", "Research"])
dream = st.selectbox("Dream Job Role", [
    "Software Developer", "Data Analyst", "AI/ML Engineer",
    "Cybersecurity Specialist", "DevOps Engineer", "Product Manager"
])

# Prediction Button 
if st.button("👉 Recommend Job Role"):
    # Create one row of input
    sample = {
        "age": age,
        "education_level": education,
        "background": background,
        "programming_languages": prog,
        "tools_known": tool,
        "interests": interest,
        "soft_skills": soft_skill,
        "study_hours_per_day": study_hours,
        "preferred_study_time": preferred_time,
        "learning_style": learning_style,
        "preferred_domain": preferred_domain,
        "dream_job_role": dream
    }

    # Convert to dataframe
    df_sample = pd.DataFrame([sample])

    # Match columns to training data
    df_encoded = pd.get_dummies(df_sample)
    df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

    # Make prediction
    pred = model.predict(df_encoded)
    result = label_encoder.inverse_transform(pred)[0]

    # Show result
    st.success(f"🎯 Recommended Job Role: *{result}*")
