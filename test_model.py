import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load test data
test_data = pd.read_csv("dataset/DailyDelhiClimateTest.csv")
test_data = test_data.rename(columns={'meantemp': 'temperature', 'meanpressure': 'pressure'})

# -------- Test Model A --------
model_A = joblib.load("model_temp_to_hwp.pkl")

X_test_A = test_data[['temperature']]
y_test_A = test_data[['humidity', 'wind_speed', 'pressure']]

preds_A = model_A.predict(X_test_A)

print(" Model A (Temp to HWP) Evaluation:")
print("MSE:", mean_squared_error(y_test_A, preds_A))
print("R2 Score:", r2_score(y_test_A, preds_A))

# -------- Test Model B --------
model_B = joblib.load("model_hwp_to_temp.pkl")

X_test_B = test_data[['humidity', 'wind_speed', 'pressure']]
y_test_B = test_data[['temperature']]

preds_B = model_B.predict(X_test_B)

print("\n Model B (HWP to Temp) Evaluation:")
print("MSE:", mean_squared_error(y_test_B, preds_B))
print("R2 Score:", r2_score(y_test_B, preds_B))


#Plot Graphs

# Plot Actual vs. Predicted for Model A
features_A = ['humidity', 'wind_speed', 'pressure']
for i, feature in enumerate(features_A):
    plt.figure()
    plt.plot(y_test_A[feature].values, label='Actual', marker='o')
    plt.plot(preds_A[:, i], label='Predicted', marker='x')
    plt.title(f'Model A: {feature} - Actual vs Predicted')
    plt.xlabel('Sample')
    plt.ylabel(feature)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# Plot Actual vs. Predicted for Model B
plt.figure()
plt.plot(y_test_B.values, label='Actual', marker='o')
plt.plot(preds_B, label='Predicted', marker='x')
plt.title('Model B: Temperature - Actual vs Predicted')
plt.xlabel('Sample')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()