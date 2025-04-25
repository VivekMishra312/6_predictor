# main.py

import os
import json
from util import generate_output

# GitHub base URL for real-time logs
BASE_URL = "https://raw.githubusercontent.com/VivekMishra312/Mumbai-aqi-logger/main/output"

# All 18 locations
LOCATIONS = [
    "Bandra_Kurla_Complex_Mumbai_-_IITM",
    "Bandra_Kurla_Complex_Mumbai_-_MPCB",
    "Borivali_East_Mumbai_-_IITM",
    "Borivali_East_Mumbai_-_MPCB",
    "Byculla_Mumbai_-_BMC",
    "Chembur_Mumbai_-_MPCB",
    "Chhatrapati_Shivaji_Intl_Airport_(T2)_Mumbai_-_MPCB",
    "Colaba_Mumbai_-_MPCB",
    "Kandivali_East_Mumbai_-_MPCB",
    "Kherwadi_Bandra_East_Mumbai_-_MPCB",
    "Khindipada-Bhandup_West_Mumbai_-_IITM",
    "Kurla_Mumbai_-_MPCB",
    "Malad_West_Mumbai_-_IITM",
    "Mazgaon_Mumbai_-_IITM",
    "Navy_Nagar-Colaba_Mumbai_-_IITM",
    "Powai_Mumbai_-_MPCB",
    "Siddharth_Nagar-Worli_Mumbai_-_IITM",
    "Vile_Parle_West_Mumbai_-_MPCB"
]

OUTPUT_DIR = "predictions"

def save_predictions(location, predictions):
    loc_folder = os.path.join(OUTPUT_DIR, location)
    os.makedirs(loc_folder, exist_ok=True)
    output_path = os.path.join(loc_folder, "prediction.json")
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)

def main():
    for loc in LOCATIONS:
        print(f"Processing {loc}...")
        result = generate_output(loc, BASE_URL)
        if result:
            save_predictions(loc, result)
            print(f"Saved predictions for {loc}")
        else:
            print(f"Skipped {loc} due to insufficient data.")

if __name__ == "__main__":
    main()
