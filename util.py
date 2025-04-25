import requests
import datetime
import numpy as np
import torch
from model import load_model
import json
import os

def fetch_last_48_hours_data(location, base_url):
    data = []
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)  # IST
    for i in range(48, 0, -1):  # last 48 hours
        ts = now - datetime.timedelta(hours=i)
        filename = ts.strftime('%Y%m%d_%H0000') + '.json'
        url = f"{base_url}/{location}/{filename}"
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                data.append(res.json())
        except Exception as e:
            print(f"Failed to fetch {url}:", e)
    return data

def get_time_features(dt):
    return [dt.hour, dt.day, dt.month, dt.weekday()]

def preprocess_data(history):
    def safe_int(val):
        try:
            return int(float(val))
        except:
            return 0

    features = []
    for entry in history:
        try:
            dt = datetime.datetime.strptime(entry['date'] + ' ' + entry['time'], "%Y-%m-%d %H:%M:%S")
            features.append(get_time_features(dt))
        except:
            continue

    recent = history[-1] if history else {}
    recent_values = [safe_int(recent.get(p, 0)) for p in ['PM2.5','PM10','NO','NO2','NOx','NH3','SO2','CO','O3']]
    
    return features, recent_values


def predict_next_6_days(model, scaler_x, scaler_y, history_data):
    features = []
    # Default standard AQI prediction values (e.g., PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, O3)
    standard_prediction = [50, 100, 20, 40, 50, 10, 5, 0.5, 40]  # Example default values

    # Preprocess available data
    for entry in history_data:
        try:
            dt = datetime.datetime.strptime(entry['date'] + ' ' + entry['time'], "%Y-%m-%d %H:%M:%S")
            features.append(get_time_features(dt))
        except:
            continue

    if len(features) < 24:
        print("Insufficient history data. Using standard AQI values.")
        # Use standard values if history data is less than 24
        features = [[0, 0, 0, 0]] * 24  # Just pad with dummy time features

    # If we have enough features, proceed with prediction
    last_24 = features[-24:]
    input_seq = torch.tensor([last_24], dtype=torch.float32)
    input_seq_scaled = scaler_x.transform(input_seq.reshape(-1, 4)).reshape(1, -1, 4)
    input_seq_scaled = torch.tensor(input_seq_scaled, dtype=torch.float32)

    preds = []
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)
    for i in range(1, 6 * 8 + 1):  # 6 days, 3-hr slots = 48 predictions
        future_dt = now + datetime.timedelta(hours=i * 3)
        time_feat = get_time_features(future_dt)
        input_window = last_24[1:] + [time_feat]
        last_24 = input_window

        input_tensor = scaler_x.transform(np.array(input_window).reshape(-1, 4)).reshape(1, -1, 4)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        
        with torch.no_grad():
            pred = model(input_tensor).numpy()
            pred = scaler_y.inverse_transform(pred).flatten()

            # Clip negative values to zero to avoid unrealistic predictions
            pred = np.clip(pred, 0, None)  # Ensure no negative predictions
            
            # Round values and convert them to integers
            pred = np.round(pred).astype(int).tolist()
            
            # Create the AQI and predominant pollutant based on predictions
            aqi = calculate_aqi(pred)  # You can use your existing logic to calculate AQI from predictions
            predominant = calculate_predominant_pollutant(pred)  # Logic to determine the predominant pollutant

            # Format output in the requested format
            formatted_output = {
                "date": future_dt.strftime("%Y-%m-%d"),
                "time": future_dt.strftime("%H:%M:%S"),
                "PM2.5": pred[0],
                "PM10": pred[1],
                "NO": "NA",  # Assuming 'NA' if no data
                "NO2": pred[3],
                "NOx": "NA",  # Assuming 'NA' if no data
                "NH3": pred[5],
                "SO2": pred[6],
                "CO": pred[7],
                "Ozone": pred[8],
                "AQI": aqi,
                "predominant": predominant
            }

            preds.append(formatted_output)

    return preds

def calculate_aqi(prediction):
    # Example logic to calculate AQI (implement this based on your domain-specific thresholds)
    # This can be a weighted average or a predefined mapping to AQI categories
    return int(np.mean(prediction))  # Just a simple average here for example

def calculate_predominant_pollutant(prediction):
    # Example logic to determine the predominant pollutant
    pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'SO2', 'CO', 'Ozone']
    return pollutants[np.argmax(prediction)]  # Return the pollutant with the highest value

def average_with_latest(predictions, latest):
    return [
        {
            'timestamp': p['timestamp'],
            'prediction': [(a + b) // 2 for a, b in zip(p['prediction'], latest)]
        }
        for p in predictions
    ]

def save_output_json(location, output_data, output_dir="predictions"):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{location}.json")
    with open(file_path, 'w') as f:
        json.dump(output_data, f, indent=2)

def generate_output(location, base_url):
    print(f"[START] Processing: {location}")
    
    try:
        print("[STEP] Fetching 48h data...")
        data_48h = fetch_last_48_hours_data(location, base_url)
        print("[OK] 48h data fetched")

        print("[STEP] Preprocessing data...")
        input_features, recent_values = preprocess_data(data_48h)
        print("[OK] Preprocessing done")

        print("[STEP] Loading model and scalers...")
        model, scaler_x, scaler_y = load_model()
        print("[OK] Model loaded")

        print("[STEP] Making predictions...")
        predictions = predict_next_6_days(input_features, model, scaler_x, scaler_y)
        print("[OK] Prediction done")

        print("[STEP] Averaging with latest value...")
        averaged = average_with_latest(predictions, recent_values)
        print("[OK] Averaging done")

        print("[STEP] Saving predictions...")
        save_output_json(location, averaged)
        print(f"[DONE] Saved predictions for {location}")

        return averaged

    except Exception as e:
        print(f"[ERROR] while processing {location}: {e}")
        return None
