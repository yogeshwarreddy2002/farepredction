# app.py - Your Final Flask Backend Server with Geocoding and Detailed Breakdown

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import shap 
import requests

# Initialize the Flask application

app = Flask(__name__)

# --- Load Model and SHAP Explainer ---

print("Loading the trained model and SHAP explainer...")
try:
    model = joblib.load('xgb_model_final.pkl')
    explainer = shap.TreeExplainer(model)
    print("Model and explainer loaded successfully.")
except FileNotFoundError:
    print("\n--- FATAL ERROR ---")
    print("Model file 'xgb_model_final.pkl' not found.")
    print("Please make sure your saved model is in the 'FairFarePredictor' folder.")
    print("-------------------\n")
    exit()

# --- Geocoding Function ---

def get_coords(location_name):
    """Converts a location name to latitude and longitude using OpenStreetMap."""
    # Add "New York" to the query to improve accuracy
    url = f"https://nominatim.openstreetmap.org/search?q={location_name}, New York&format=json"
    # It's good practice to set a custom User-Agent for API calls
    headers = {'User-Agent': 'FairFareAI/1.0'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    if data:
        # Return the coordinates of the first result
        return {"latitude": float(data[0]["lat"]), "longitude": float(data[0]["lon"])}
    return None

# --- Feature Engineering Function (Matches your notebook) ---

def prepare_features(ride_details):
    df = pd.DataFrame([ride_details])
    pickup_datetime = datetime.now()
    df['pickup_datetime_year'] = pickup_datetime.year
    df['pickup_datetime_month'] = pickup_datetime.month
    df['pickup_datetime_day'] = pickup_datetime.day
    df['pickup_datetime_weekday'] = pickup_datetime.weekday()
    df['pickup_datetime_hour'] = pickup_datetime.hour

    lon1, lat1, lon2, lat2 = map(np.radians, [df['pickup_longitude'], df['pickup_latitude'], df['dropoff_longitude'], df['dropoff_latitude']])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df['trip_distance'] = 6371 * c

    landmarks = {
        'jfk': (-73.7781, 40.6413), 'lga': (-73.8740, 40.7769), 'ewr': (-74.1745, 40.6895),
        'met': (-73.9632, 40.7794), 'wtc': (-74.0099, 40.7126)
    }
    for name, (lon, lat) in landmarks.items():
        lon1_land, lat1_land, lon2_land, lat2_land = map(np.radians, [lon, lat, df['dropoff_longitude'], df['dropoff_latitude']])
        dlon_land = lon2_land - lon1_land
        dlat_land = lat2_land - lat1_land
        a_land = np.sin(dlat_land/2.0)**2 + np.cos(lat1_land) * np.cos(lat2_land) * np.sin(dlon_land/2.0)**2
        c_land = 2 * np.arcsin(np.sqrt(a_land))
        df[name + '_drop_distance'] = 6371 * c_land
    
    feature_cols = [
        'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
        'passenger_count', 'pickup_datetime_year', 'pickup_datetime_month',
        'pickup_datetime_day', 'pickup_datetime_weekday', 'pickup_datetime_hour',
        'trip_distance', 'jfk_drop_distance', 'lga_drop_distance',
        'ewr_drop_distance', 'met_drop_distance', 'wtc_drop_distance'
    ]
    return df[feature_cols]

# --- API Routes ---

@app.route('/fare-prediction-nyc')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        pickup_name = data['pickup_location']
        dropoff_name = data['dropoff_location']
        
        pickup_coords = get_coords(pickup_name)
        dropoff_coords = get_coords(dropoff_name)

        if not pickup_coords or not dropoff_coords:
            return jsonify({'error': 'Could not find one or both locations. Please be more specific.'}), 400

        ride_details = {
            'pickup_latitude': pickup_coords['latitude'], 'pickup_longitude': pickup_coords['longitude'],
            'dropoff_latitude': dropoff_coords['latitude'], 'dropoff_longitude': dropoff_coords['longitude'],
            'passenger_count': int(data['passenger_count'])
        }

        features = prepare_features(ride_details)
        prediction = float(model.predict(features)[0])
        
        shap_values = explainer.shap_values(features)[0]
        
        base_fare = float(explainer.expected_value)
        
        # Categorize SHAP values into the requested components

        distance_cost = float(shap_values[features.columns.get_loc('trip_distance')])
        time_cost = float(shap_values[features.columns.get_loc('pickup_datetime_hour')])
        day_cost = float(shap_values[features.columns.get_loc('pickup_datetime_weekday')])
        
        # Bundle all other small feature contributions into the base fare to ensure the total adds up
        other_costs = prediction - base_fare - distance_cost - time_cost - day_cost
        final_base_fare = base_fare + other_costs

        response_data = {
            "predicted_fare": round(prediction, 2),
            "base_fare": round(final_base_fare, 2),
            "distance_cost": round(distance_cost, 2),
            "time_cost": round(time_cost, 2),
            "day_cost": round(day_cost, 2)
        }
        
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
