from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
import pickle

# Load the model and encoders
model = joblib.load(r'C:\Users\syeda\OneDrive\Desktop\Flight delays prediction\flight_model.pkl')
encoders = joblib.load(r'C:\Users\syeda\OneDrive\Desktop\Flight delays prediction\encoders.pkl')


df = pd.read_csv(r"C:\Users\syeda\OneDrive\Desktop\Flight delays prediction\flight_data.csv")
#df = pd.read_csv("flight_data.csv")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get the values from the form
        carrier = request.form['carrier']
        origin = request.form['origin']
        dest = request.form['dest']
        distance = int(request.form['distance'])
        hour = int(request.form['hour'])

        day = int(request.form['day'])
        month = int(request.form['month'])

        # Prepare the input data
        input_data = pd.DataFrame({
            'carrier': [carrier],
            'origin': [origin],
            'dest': [dest],
            'distance': [distance],
            'hour': [hour],
            'day': [day],
            'month': [month]
        })

        # Apply encoders
        for col in encoders.keys():
            input_data[col] = encoders[col].transform(input_data[col])[0]

        # Make the prediction
        prediction = model.predict(input_data.values)
        result = 'This flight is likely to be departing late. Thank You for your Cooperation.' if prediction[0] == 1 else 'This flight is likely to be departing on time.'
        
        return render_template('index.html', result=result, carrier=carrier, origin=origin, dest=dest, distance=distance, hour=hour, day=day, month=month,
                               carriers=df['carrier'].unique(), origins=df['origin'].unique(), destinations=df['dest'].unique())

    return render_template('index.html', result=None, carriers=df['carrier'].unique(), origins=df['origin'].unique(), destinations=df['dest'].unique())

if __name__ == "__main__":
    app.run(debug=True)
