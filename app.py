from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load pre-trained models
model_A = joblib.load("model_temp_to_hwp.pkl")  # Temperature ➜ Humidity, Wind Speed, Pressure
model_B = joblib.load("model_hwp_to_temp.pkl")  # Humidity, Wind Speed, Pressure ➜ Temperature

@app.route('/', methods=['GET', 'POST'])
def index():
    result = {}

    if request.method == 'POST':
        mode = request.form.get('mode')
        result['mode'] = mode

        try:
            if mode == 'forward':
                # Temperature ➜ H, W, P
                temperature = float(request.form.get('temperature'))
                prediction = model_A.predict([[temperature]])[0]
                result['temperature'] = temperature
                result['humidity'] = float(prediction[0])
                result['wind_speed'] = float(prediction[1])
                result['pressure'] = float(prediction[2])

            elif mode == 'reverse':
                # H, W, P ➜ Temperature
                humidity = float(request.form.get('humidity'))
                wind_speed = float(request.form.get('wind_speed'))
                pressure = float(request.form.get('pressure'))
                prediction = model_B.predict([[humidity, wind_speed, pressure]])[0]
                result['humidity'] = humidity
                result['wind_speed'] = wind_speed
                result['pressure'] = pressure
                result['temperature'] = float(prediction)

        except Exception as e:
            result['error'] = f" Error in input: {str(e)}"

    return render_template("index.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)
