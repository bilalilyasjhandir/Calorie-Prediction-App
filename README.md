# 🔥 Calorie Burn Prediction App

A machine learning web app built with **Streamlit** that predicts calories burned during a workout using a trained **LightGBM** regression model.

## Prerequisites

- Python 3.8 or higher
- The following files must be in the project directory:
  - `calorie_model_tuned.pkl` — trained LightGBM model
  - `scaler.pkl` — fitted scaler

## Installation

1. **Clone or download** this repository.

2. **Install dependencies:**

   ```bash
   pip install streamlit pandas joblib lightgbm scikit-learn
   ```

## Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

## Usage

1. Enter your details: age, gender, height, weight, workout duration, heart rate, and body temperature.
2. Click **Predict Calories**.
3. View the estimated calories burned on screen.

## Project Structure

```
Fitness-App/
├── app.py                      # streamlit web application
├── calorie_model_tuned.pkl     # trained lightgbm model
├── scaler.pkl                  # fitted scaler
└── README.md                   # project documentation
```
