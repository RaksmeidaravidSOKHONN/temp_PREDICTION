import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# -------------------- Load and preprocess test data -------------------- #
test_data = pd.read_csv("dataset/DailyDelhiClimateTest.csv")
test_data = test_data.rename(columns={'meantemp': 'temperature', 'meanpressure': 'pressure'})

# Remove pressure outliers
test_data = test_data[test_data['pressure'].between(900, 1100)]

# Date features
test_data['date'] = pd.to_datetime(test_data['date'])
test_data['month'] = test_data['date'].dt.month
test_data['day_of_year'] = test_data['date'].dt.dayofyear
test_data['season'] = test_data['month'] % 12 // 3 + 1

# Drop date column
test_data = test_data.drop(columns=['date'])

# -------------------- Model A: Temp + Time ➜ H, W, P -------------------- #
features_A = ['temperature', 'month', 'day_of_year', 'season']
X_test_A = test_data[features_A]
y_test_A = test_data[['humidity', 'wind_speed', 'pressure']]

# Load models
model_A1 = joblib.load("model_temp_to_humidity.pkl")
model_A2 = joblib.load("model_temp_to_wind.pkl")
model_A3 = joblib.load("model_temp_to_pressure.pkl")

# Predict
pred_humidity = model_A1.predict(X_test_A)
pred_wind = model_A2.predict(X_test_A)
pred_pressure = model_A3.predict(X_test_A)

# Stack predictions
preds_A = np.column_stack([pred_humidity, pred_wind, pred_pressure])

# Model A Evaluation 
mse_A = mean_squared_error(y_test_A, preds_A)
rmse_A = np.sqrt(mse_A)
mae_A = mean_absolute_error(y_test_A, preds_A)
r2_A = r2_score(y_test_A, preds_A)

print("Model A (Temperature -> Humidity, Wind, Pressure):")
print(f"  MSE : {mse_A:.2f}")
print(f"  RMSE: {rmse_A:.2f}")
print(f"  MAE : {mae_A:.2f}")
print(f"  R^2  : {r2_A:.3f}")

# -------------------- Model B: H, W, P ➜ Temp -------------------- #
features_B = ['humidity', 'wind_speed', 'pressure']
X_test_B = test_data[features_B]
y_test_B = test_data['temperature']

model_B = joblib.load("model_hwp_to_temp.pkl")
preds_B = model_B.predict(X_test_B)

# Model B Evaluation
mse_B = mean_squared_error(y_test_B, preds_B)
rmse_B = np.sqrt(mse_B)
mae_B = mean_absolute_error(y_test_B, preds_B)
r2_B = r2_score(y_test_B, preds_B)

print("\nModel B (Humidity, Wind, Pressure -> Temperature):")
print(f"  MSE : {mse_B:.2f}")
print(f"  RMSE: {rmse_B:.2f}")
print(f"  MAE : {mae_B:.2f}")
print(f"  R^2  : {r2_B:.3f}")

