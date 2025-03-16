# Numeric Weather Prediction with Ensemble Kalman Filter (EnKF)

## 🌦 Project Overview
This project focuses on **short-term weather forecasting** using **Numeric Weather Prediction (NWP)** with **data assimilation**. The **Ensemble Kalman Filter (EnKF)** is used to integrate **publicly available local weather station data** and predict key weather parameters, such as temperature, precipitation, humidity, and wind speed, for the next three days.

## 📌 Features
- **Data Assimilation with EnKF**: Improves forecast accuracy by integrating historical weather data.
- **Weather Parameter Forecasting**: Predicts temperature, precipitation, wind speed, and more.
- **Visualization & Evaluation**:
  - Scatter plots (Wind Speed vs. Cloud Cover)
  - Error distribution analysis
  - Actual vs. Predicted weather plots
- **Forecasting**: Generates a 3-day future weather prediction.

## 📂 Dataset Details
The dataset includes the following columns:
```
- date
- temperature_2m
- relative_humidity_2m
- dew_point_2m
- apparent_temperature
- precipitation
- rain
- snowfall
- snow_depth
- pressure_msl
- surface_pressure
- cloud_cover
- cloud_cover_low
- cloud_cover_mid
- cloud_cover_high
- wind_speed_10m
```

## 🛠 Tech Stack
- **Python** (NumPy, Pandas, Matplotlib, Seaborn, SciPy)
- **FilterPy** (for EnKF implementation)
- **TQDM** (progress tracking)
- **Jupyter Notebook / Google Colab**

## 🔧 Installation & Setup
```bash
# Clone this repository
git clone https://github.com/22danu/Weather

# Install dependencies
pip install numpy pandas matplotlib seaborn scipy tqdm filterpy

# Run the Jupyter Notebook
jupyter notebook
```

## 🚀 Usage
1. **Load the dataset** and preprocess it.
2. **Initialize the Ensemble Kalman Filter (EnKF)**.
3. **Run the EnKF prediction loop** to update forecasts.
4. **Visualize predictions** and compare actual vs. predicted weather parameters.
5. **Generate a 3-day forecast** and analyze trends.

## 📊 Evaluation Metrics
- **Confusion Matrix** (for classification-based predictions)
- **Accuracy Score**
- **Error Distribution Analysis**
- **RMSE (Root Mean Squared Error)**
- **Visualization of Actual vs. Predicted values**

## 📌 Example Visualizations
### 🌬 Wind Speed vs. Cloud Cover
```python
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x="wind_speed_10m", y="cloud_cover", hue="precipitation", palette="coolwarm")
plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Cloud Cover (%)")
plt.title("Wind Speed vs Cloud Cover")
plt.show()
```

### 📈 3-Day Weather Forecast (Temperature & Precipitation)
```python
plt.figure(figsize=(12, 6))
plt.plot(days, future_predictions[:, temp_idx], label="Predicted Temperature", color='red', linestyle='dashed')
plt.xlabel("Days from Now")
plt.ylabel("Temperature (°C)")
plt.title("3-Day Weather Forecast - Temperature")
plt.legend()
plt.show()
```

## 🤝 Contributing
Feel free to contribute by improving the data preprocessing, optimizing the EnKF algorithm, or adding new features!

## 📜 License
This project is **open-source** under the MIT License.

---

💡 **Author**: Vanitha A, Danushiyaa M