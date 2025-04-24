import os
import json
import random
import requests
from datetime import datetime, timedelta
from collections import defaultdict

# Configurations
LOCATIONS = [
    "Bandra_Kurla_Complex_Mumbai_-_IITM", "Bandra_Kurla_Complex_Mumbai_-_MPCB",
    "Borivali_East_Mumbai_-_IITM", "Borivali_East_Mumbai_-_MPCB",
    "Byculla_Mumbai_-_BMC", "Chembur_Mumbai_-_MPCB",
    "Chhatrapati_Shivaji_Intl_Airport_(T2)_Mumbai_-_MPCB", "Colaba_Mumbai_-_MPCB",
    "Kandivali_East_Mumbai_-_MPCB", "Kherwadi_Bandra_East_Mumbai_-_MPCB",
    "Khindipada-Bhandup_West_Mumbai_-_IITM", "Kurla_Mumbai_-_MPCB",
    "Malad_West_Mumbai_-_IITM", "Mazgaon_Mumbai_-_IITM",
    "Navy_Nagar-Colaba_Mumbai_-_IITM", "Powai_Mumbai_-_MPCB",
    "Siddharth_Nagar-Worli_Mumbai_-_IITM", "Vile_Parle_West_Mumbai_-_MPCB"
]

POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "SO2", "CO", "Ozone"]

SLOT_HOURS = [
    (0, 3), (3, 6), (6, 9), (9, 12),
    (12, 15), (15, 18), (18, 21), (21, 24)
]

GITHUB_BASE = "https://raw.githubusercontent.com/VivekMishra312/Mumbai-aqi-logger/main/output"
OUTPUT_DIR = "prediction"

def get_json(location, date: datetime, hour: int):
    dt_str = date.strftime("%Y%m%d") + f"_{hour:02}0000"
    url = f"{GITHUB_BASE}/{location}/{dt_str}.json"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            return json.loads(res.text)
    except:
        pass
    return None

def fetch_slot_avg(location, date: datetime, slot_start: int, slot_end: int, prev_days_data: list, apply_variation=False):
    values = defaultdict(list)

    # Fetching data from the last 3 days (including the current day)
    for day_offset in range(1, 4):  # Last 3 days (Including the current day in the loop)
        d = date - timedelta(days=day_offset)
        for hour in range(slot_start, slot_end):
            data = get_json(location, d, hour)
            if data:
                for p in POLLUTANTS:
                    v = data.get(p)
                    if v != "NA" and v is not None:
                        values[p].append(float(v))

    # Add the previous days' predictions into the averaging
    for past_day in prev_days_data[-3:]:  # Last 3 days of predicted values
        for p in POLLUTANTS:
            if past_day.get(p):
                values[p].append(past_day[p])

    # Calculate the average of the values collected
    slot_avg = {}
    for p in POLLUTANTS:
        if values[p]:
            avg = sum(values[p]) / len(values[p])
            if apply_variation:  # Apply 10% variation if required
                variation = 0.1 * avg  # 10% variation
                avg += random.choice([-variation, variation])  # Randomly apply +10% or -10%
            slot_avg[p] = round(avg)  # Round to nearest integer
            print(f"Averaging for {p}: {values[p]} => {slot_avg[p]}")
    return slot_avg

def predict_slots(location):
    now = datetime.utcnow() + timedelta(hours=5, minutes=30)  # IST
    base_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    predictions = []

    prev_days_data = []

    for day in range(6):  # Predict for the next 6 days
        day_date = base_date + timedelta(days=day)
        day_str = day_date.strftime("%Y-%m-%d")

        for slot_start, slot_end in SLOT_HOURS:
            apply_variation = False
            if day >= 3:  # For Day 4, 5, 6, we apply 10% variation
                apply_variation = True

            avg = fetch_slot_avg(location, day_date, slot_start, slot_end, prev_days_data, apply_variation)

            # Include the slot data
            final = {p: avg.get(p, "NA") for p in POLLUTANTS}
            final["date"] = day_str
            final["slot"] = f"{slot_start:02}:00-{slot_end:02}:00"

            predictions.append(final)

        # Store current day's prediction for future reference
        prev_days_data.append({p: avg.get(p, None) for p in POLLUTANTS})

    return predictions

def save_predictions(location, data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{location}.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    for loc in LOCATIONS:
        print(f"Processing {loc}...")
        data = predict_slots(loc)
        save_predictions(loc, data)

if __name__ == "__main__":
    main()
