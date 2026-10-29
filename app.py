
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Student Pass Prediction",
    page_icon="🎓",
    layout="centered"
)

# App title
st.title("🎓 Student Pass Prediction")
st.write("Enter the student's details to predict whether they will pass.")

# Load trained model
model_path = Path(__file__).resolve().parent / "logisticRegression.pkl"

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("model.pkl not found. Place it in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.subheader("Student Details")

# Input 1: Student ID
student_id = st.number_input(
    "Student ID",
    min_value=0,
    value=1,
    step=1
)

# Input 2: Study hours per day
study_hours = st.number_input(
    "Study Hours Per Day",
    min_value=0.0,
    max_value=24.0,
    value=4.0,
    step=0.5
)

# Input 3: Attendance percentage
attendance = st.number_input(
    "Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

# Input 4: Sleep hours
sleep_hours = st.number_input(
    "Sleep Hours Per Day",
    min_value=0.0,
    max_value=24.0,
    value=8.0,
    step=0.5
)

# Input 5: Previous exam score
previous_score = st.number_input(
    "Previous Exam Score",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=1.0
)

# Input 6: Practice tests completed
practice_tests = st.number_input(
    "Practice Tests Completed",
    min_value=0,
    value=5,
    step=1
)

# Prediction button
if st.button("Predict Result", type="primary"):

    # Keep feature names consistent with model training
    input_data = pd.DataFrame([{
        "student_id": student_id,
        "study_hours_per_day": study_hours,
        "attendance_percent": attendance,
        "sleep_hours": sleep_hours,
        "previous_exam_score": previous_score,
        "practice_tests_completed": practice_tests
    }])

    try:
        prediction = float(model.predict(input_data)[0])

        st.subheader("Prediction Result")
        st.write(f"Raw model prediction: {prediction:.2f}")

        # Assumes passed is encoded as 0 = Fail and 1 = Pass
        if prediction >= 0.5:
            st.success("pass")
        else:
            st.error("fail")

        st.caption(
            "This prediction is an estimate, not a guaranteed result."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Make sure the model was trained with these exact six "
            "features and compatible column names."
        )
