from flask import Flask, render_template, request
import joblib
from datetime import datetime

app = Flask(__name__)

# Load no-lag models
model_A1 = joblib.load("model_temp_to_humidity.pkl")
model_A2 = joblib.load("model_temp_to_wind.pkl")
model_A3 = joblib.load("model_temp_to_pressure.pkl")
model_B = joblib.load("model_hwp_to_temp.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = {}

    if request.method == 'POST':
        mode = request.form.get('mode')
        result['mode'] = mode

        try:
            if mode == 'forward':
                # Temp ➜ Humidity, Wind Speed, Pressure
                temperature = float(request.form.get('temperature'))
                date_str = request.form.get('date')  # Format: YYYY-MM-DD
                date = datetime.strptime(date_str, '%Y-%m-%d')
                month = date.month
                day_of_year = date.timetuple().tm_yday
                season = (month % 12) // 3 + 1

                features = [[temperature, month, day_of_year, season]]

                humidity = model_A1.predict(features)[0]
                wind_speed = model_A2.predict(features)[0]
                pressure = model_A3.predict(features)[0]

                result.update({
                    'temperature': temperature,
                    'date': date_str,
                    'humidity': round(humidity, 2),
                    'wind_speed': round(wind_speed, 2),
                    'pressure': round(pressure, 2)
                })

            elif mode == 'reverse':
                # H, W, P ➜ Temp
                humidity = float(request.form.get('humidity'))
                wind_speed = float(request.form.get('wind_speed'))
                pressure = float(request.form.get('pressure'))

                features = [[humidity, wind_speed, pressure]]
                temperature = model_B.predict(features)[0]

                result.update({
                    'humidity': humidity,
                    'wind_speed': wind_speed,
                    'pressure': pressure,
                    'temperature': round(temperature, 2),
                    'date': request.form.get('date')
                })

        except Exception as e:
            result['error'] = f"⚠️ Error in input: {str(e)}"

    return render_template("index.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)
