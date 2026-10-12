import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for shadow and card styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Load the Model
@st.cache_resource
def load_model():
  try:
    with open("kn.pkl", "rb") as f:
      model = pickle.load(f)
    return model
  except Exception as e:
    st.error(f"Error loading model file 'kn.pkl': {e}")
    return None


model = load_model()

# App Header
st.title("🎓 Student Performance & Outcome Predictor")
st.markdown(
    "Provide the student details in the sidebar to generate performance"
    " predictions using your trained KNeighbors Regressor model."
)

# Sidebar Inputs Form
st.sidebar.header("📝 Student Parameters")

with st.sidebar:
  student_id = st.text_input("Student ID", value="STU1001")
  age = st.slider("Age", min_value=10, max_value=60, value=20)
  gender = st.selectbox("Gender", ["Male", "Female", "Other"])
  course = st.selectbox(
      "Course", ["Computer Science", "Engineering", "Business", "Arts", "Science"]
  )
  study_hours = st.slider(
      "Study Hours (Weekly)", min_value=0.0, max_value=50.0, value=15.0, step=0.5
  )
  class_attendance = st.slider(
      "Class Attendance (%)", min_value=0.0, max_value=100.0, value=85.0
  )
  internet_access = st.selectbox("Internet Access", ["Yes", "No"])
  sleep_hours = st.slider(
      "Sleep Hours (Daily)", min_value=0.0, max_value=12.0, value=7.0, step=0.5
  )
  sleep_quality = st.slider("Sleep Quality Rating (1-10)", 1, 10, 7)
  study_method = st.selectbox(
      "Study Method", ["Self-Study", "Group Study", "Online Courses", "Hybrid"]
  )
  facility_rating = st.slider("Facility Rating (1-5)", 1, 5, 4)
  exam_difficulty = st.selectbox("Exam Difficulty", ["Easy", "Medium", "Hard"])

# Main Dashboard Layout
col1, col2 = st.columns([2, 1])

with col1:
  st.markdown(
      """
        <div class="card">
            <h3>📊 Input Overview</h3>
            <p>Review your selected configurations before running the prediction model.</p>
        </div>
        """,
      unsafe_allow_html=True,
  )

  # Display inputs neatly in a table
  input_data_df = pd.DataFrame({
      "Feature": [
          "Student ID",
          "Age",
          "Gender",
          "Course",
          "Study Hours",
          "Attendance",
          "Internet",
          "Sleep Hours",
          "Sleep Quality",
          "Study Method",
          "Facility Rating",
          "Exam Difficulty",
      ],
      "Value": [
          student_id,
          age,
          gender,
          course,
          study_hours,
          class_attendance,
          internet_access,
          sleep_hours,
          sleep_quality,
          study_method,
          facility_rating,
      ],
  })
  st.dataframe(input_data_df, use_container_width=True)

with col2:
  st.markdown(
      """
        <div class="card">
            <h3>🚀 Prediction</h3>
        </div>
        """,
      unsafe_allow_html=True,
  )

  if st.button("Predict Outcome"):
    if model is not None:
      try:
        # Note: If your model was trained with encoded categorical features (OrdinalEncoder/OneHotEncoder),
        # make sure to transform these inputs accordingly. Below is a template mapping numeric conversions:
        gender_val = 1 if gender == "Male" else (0 if gender == "Female" else 2)
        internet_val = 1 if internet_access == "Yes" else 0

        # Create numerical feature vector matching your training columns:
        # ['student_id', 'age', 'gender', 'course', 'study_hours', 'class_attendance',
        #  'internet_access', 'sleep_hours', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']
        # (Adjust encoding mappings based on how your scikit-learn model was originally trained)

        # Placeholder numeric representation for demo execution:
        features = np.array([[
            hash(student_id) % 10000,
            age,
            gender_val,
            hash(course) % 100,
            study_hours,
            class_attendance,
            internet_val,
            sleep_hours,
            sleep_quality,
            hash(study_method) % 100,
            facility_rating,
            (
                1
                if exam_difficulty == "Easy"
                else (2 if exam_difficulty == "Medium" else 3)
            ),
        ]])

        prediction = model.predict(features)
        st.success(f"### Predicted Result: {prediction[0]:.2f}")

      except Exception as ex:
        st.error(
            "Prediction failed. Ensure your categorical variables match the"
            f" exact preprocessing used during training. Details: {ex}"
        )
    else:
      st.warning("Model is not loaded correctly.")
