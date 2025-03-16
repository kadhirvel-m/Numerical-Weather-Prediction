import numpy as np
import pandas as pd
import streamlit as st
from filterpy.kalman import EnsembleKalmanFilter as EnKF

# Define state transition and observation functions
def fx(x, dt):
    return x  

def hx(x):
    return x  

# Function to Load Trained Weights
def load_enkf_weights(enkf, filename="enkf_weights.npz"):
    data = np.load(filename)
    enkf.x = data["x"]
    enkf.P = data["P"]
    st.success(f"🔄 Loaded trained weights from {filename}")

# Load dataset
df = pd.read_csv("cleaned_weather_data.csv", parse_dates=["date"])
state_variables = ["temperature_2m", "wind_speed_10m", "cloud_cover", "precipitation"]
df = df[state_variables + ["date"]].dropna()

# Initialize EnKF
num_ensemble = 100
state_dim = len(state_variables)
dt = 1  # Time step

enkf = EnKF(x=np.zeros(state_dim), P=np.eye(state_dim), N=num_ensemble, dim_z=state_dim, dt=dt, hx=hx, fx=fx)

# Load trained weights
load_enkf_weights(enkf)

# Function to Predict Weather for Multiple Days
def predict_weather(target_date, days=3):
    predictions = []
    target_date = pd.to_datetime(target_date).tz_localize("UTC")
    
    for i in range(days):
        date = target_date + pd.Timedelta(days=i)
        closest_data = df[df["date"] <= date].iloc[-1]
        obs = closest_data[state_variables].astype(float).values
        
        enkf.predict()
        enkf.update(obs)
        
        predicted_state = enkf.x
        predictions.append({var: pred for var, pred in zip(state_variables, predicted_state)})
        predictions[-1]["date"] = date.strftime("%Y-%m-%d")
    
    return predictions

# Streamlit UI
st.title("🌤️ AI-Powered Weather Forecast")
st.write("Predicting short-term weather using Ensemble Kalman Filter")

target_date = st.date_input("Select Start Date for Prediction", pd.to_datetime("2025-03-20"))

if st.button("Generate Forecast"):
    forecast = predict_weather(target_date, days=3)
    
    # Display predictions
    st.subheader("📅 3-Day Weather Forecast")
    for day in forecast:
        with st.container():
            st.markdown(f"### {day['date']}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🌡️ Temp (°C)", f"{day['temperature_2m']:.2f}")
            col2.metric("💨 Wind Speed (m/s)", f"{day['wind_speed_10m']:.2f}")
            col3.metric("☁️ Cloud Cover (%)", f"{day['cloud_cover']:.2f}")
            col4.metric("🌧️ Precipitation (mm)", f"{day['precipitation']:.2f}")
