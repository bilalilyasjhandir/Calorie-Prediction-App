import streamlit as st
import pandas as pd
import joblib

#page configuration
st.set_page_config(
    page_title="Calorie Burn Prediction",
    page_icon="🔥",
    layout="centered"
)

#custom styling
st.markdown("""
<style>
    .main-title {
        text-align: center !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #FF4B4B !important;
        margin-bottom: 0 !important;
    }
    .sub-text {
        text-align: center !important;
        font-size: 1.2rem !important;
        color: #888 !important;
        margin-bottom: 2rem !important;
    }
    .result-box {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8C42 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        color: white;
        margin-top: 1.5rem;
    }
    .result-label {
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 0.5rem;
    }
    .result-value {
        font-size: 3rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

#title and description
st.markdown('<h1 class="main-title">Calorie Burn Prediction App</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-text">Enter your workout details below and get an instant prediction of calories burned.</p>',
    unsafe_allow_html=True
)
st.markdown("---")

#load model and scaler once using streamlit cache
@st.cache_resource
def load_model_and_scaler():
    """load the trained lightgbm model and scaler from disk."""
    model = joblib.load("calorie_model_tuned.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

#input section
st.subheader("Enter Your Details")

#arrange inputs in two columns for a neat layout
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=25, step=1)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1)
with col2:
    duration = st.number_input("Duration (minutes)", min_value=1.0, max_value=300.0, value=30.0, step=1.0)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40.0, max_value=220.0, value=100.0, step=1.0)
    body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)
st.markdown("---")

#predict button centered
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("Predict Calories", use_container_width=True, type="primary")

#run prediction when button clicked
if predict_clicked:
    try:
        model, scaler = load_model_and_scaler()
        gender_encoded = 1 if gender == "Male" else 0
        input_data = pd.DataFrame({
            "Gender": [gender_encoded],
            "Age": [age],
            "Height": [height],
            "Weight": [weight],
            "Duration": [duration],
            "Heart_Rate": [heart_rate],
            "Body_Temp": [body_temp]
        })
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        calories = round(prediction[0], 2)
        st.success("Prediction completed successfully!")
        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">Estimated Calories Burned</div>
                <div class="result-value">{calories} kcal</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("")
        st.metric(label="Calories Burned", value=f"{calories} kcal")

    except FileNotFoundError:
        st.error("Model or scaler file not found. Please make sure `calorie_model_tuned.pkl` and `scaler.pkl` are in the same directory as `app.py`.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")