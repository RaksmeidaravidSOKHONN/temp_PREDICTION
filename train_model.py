import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# ------------------ Load and Clean Data ------------------ #
df = pd.read_csv("dataset/DailyDelhiClimateTrain.csv")
df = df.rename(columns={'meantemp': 'temperature', 'meanpressure': 'pressure'})

# Remove pressure outliers
df = df[df['pressure'].between(900, 1100)]

# Date parsing
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear
df['season'] = df['month'] % 12 // 3 + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
df.drop(columns=['date'], inplace=True)

# ------------------ Split Dataset ------------------ #
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

# ------------------ Model A: Temp + Time ➜ H, W, P ------------------ #
features_A = ['temperature', 'month', 'day_of_year', 'season']
X_A = train_df[features_A]
y_humidity = train_df['humidity']
y_wind = train_df['wind_speed']
y_pressure = train_df['pressure']

model_A1 = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
model_A1.fit(X_A, y_humidity)
joblib.dump(model_A1, "model_temp_to_humidity.pkl")

model_A2 = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
model_A2.fit(X_A, y_wind)
joblib.dump(model_A2, "model_temp_to_wind.pkl")

model_A3 = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
model_A3.fit(X_A, y_pressure)
joblib.dump(model_A3, "model_temp_to_pressure.pkl")

# ------------------ Model B: H, W, P ➜ Temp ------------------ #
features_B = ['humidity', 'wind_speed', 'pressure']
X_B = train_df[features_B]
y_B = train_df['temperature']

model_B = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
model_B.fit(X_B, y_B)
joblib.dump(model_B, "model_hwp_to_temp.pkl")

print("Both models trained and saved successfully.")
