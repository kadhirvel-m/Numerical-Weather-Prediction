import numpy as np
import pandas as pd
from tqdm import tqdm
from filterpy.kalman import EnsembleKalmanFilter as EnKF

# Load dataset
df = pd.read_csv("cleaned_weather_data.csv", parse_dates=["date"])
state_variables = ["temperature_2m", "wind_speed_10m", "cloud_cover", "precipitation"]
df = df[state_variables].dropna()
observations = df.values

# Define EnKF Model
num_ensemble = 100
state_dim = observations.shape[1]

def fx(x, dt):
    """State transition function (identity for now)"""
    return x  

def hx(x):
    """Observation function (identity)"""
    return x  

dt = 1  # Time step

# Initialize EnKF
enkf = EnKF(x=np.mean(observations, axis=0), P=np.cov(observations.T), N=num_ensemble, dim_z=state_dim, dt=dt, hx=hx, fx=fx)
enkf.R *= 0.1  
enkf.Q *= 0.01  

# Run EnKF Assimilation
predicted_states = []
for obs in tqdm(observations, desc="Running EnKF Assimilation", unit="step"):
    enkf.predict()
    enkf.update(obs)
    predicted_states.append(enkf.x.copy())

# Convert predictions to DataFrame
predicted_df = pd.DataFrame(predicted_states, columns=state_variables)
predicted_df["date"] = df.index  

# **Function to Save Trained Weights (x and P only)**
def save_enkf_weights(enkf, filename="enkf_weights.npz"):
    np.savez(filename, x=enkf.x, P=enkf.P)
    print(f"✅ Trained weights saved to {filename}")

# **Function to Load Trained Weights**
def load_enkf_weights(enkf, filename="enkf_weights.npz"):
    data = np.load(filename)
    enkf.x = data["x"]
    enkf.P = data["P"]
    print(f"🔄 Loaded trained weights from {filename}")

# Save weights after training
save_enkf_weights(enkf)

# Example: Load weights when needed
# load_enkf_weights(enkf)
