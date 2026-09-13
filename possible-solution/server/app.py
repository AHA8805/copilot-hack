# Create a base Flask server

import csv
import pickle
from functools import lru_cache

from flask import Flask, jsonify, request

app = Flask(__name__)

# Enable cors
@app.after_request
def after_request(response):
    """
    Enable CORS
    """
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response


# Load model from pickle file once at startup instead of once per request.
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@lru_cache(maxsize=1)
def get_airports():
    """Load and sort airport metadata once so repeated requests avoid re-reading the CSV."""
    with open('airports.csv', newline='', encoding='utf-8') as airport_file:
        airports = [
            {'id': int(row['OriginAirportID']), 'name': row['OriginAirportName']}
            for row in csv.DictReader(airport_file)
        ]
    return sorted(airports, key=lambda airport: airport['name'])


# Model takes two parameters - day of week and airport id, then returns a prediction of flight delay
@app.route('/predict', methods=['GET'])
def predict():
    """
    Takes two parameters - day of week and airport id, then returns a prediction of flight delay
    """
    day_of_week = request.args.get('day_of_week', type=int)
    airport_id = request.args.get('airport_id', type=int)

    if day_of_week is None or airport_id is None:
        return jsonify({'error': 'day_of_week and airport_id are required'}), 400

    prediction = model.predict_proba([[day_of_week, airport_id]])[0]

    # Split prediction string by space
    prediction = str(prediction).split(' ')

    # store first value from prediction as certainty, and remove the first character
    certainty = float(prediction[0][2:])

    # store second value from prediction as delay, and remove the last character
    delay = float(prediction[1][:-1])

    # return prediction as json
    return jsonify({'certainty': certainty, 'delay': delay})


# Create a new route called airports with method of get
@app.route('/airports', methods=['GET'])
def airports():
    return jsonify(get_airports())


if __name__ == '__main__':
    app.run(debug=True, threaded=True)