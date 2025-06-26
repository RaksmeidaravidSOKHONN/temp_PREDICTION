import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import joblib

# Load training data
train_data = pd.read_csv("dataset/DailyDelhiClimateTrain.csv")

# Rename columns for consistency
train_data = train_data.rename(columns={'meantemp': 'temperature', 'meanpressure': 'pressure'})

# Model A: temperature ➜ humidity, wind_speed, pressure
X_A = train_data[['temperature']]
y_A = train_data[['humidity', 'wind_speed', 'pressure']]

model_A = MultiOutputRegressor(LinearRegression())
model_A.fit(X_A, y_A)
joblib.dump(model_A, "model_temp_to_hwp.pkl")

# Model B: humidity, wind_speed, pressure ➜ temperature
X_B = train_data[['humidity', 'wind_speed', 'pressure']]
y_B = train_data[['temperature']]

model_B = LinearRegression()
model_B.fit(X_B, y_B)
joblib.dump(model_B, "model_hwp_to_temp.pkl")

print("✅ Models trained and saved successfully.")
