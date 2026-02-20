import csv
import os
from datetime import datetime

CSV_PATH = "logs/predictions.csv"

def log_prediction(image_name, label, confidence, class_probs):
    os.makedirs("logs", exist_ok=True)

    file_exists = os.path.exists(CSV_PATH)

    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "image_name",
                "predicted_label",
                "confidence"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_name,
            label,
            round(float(confidence), 2)
        ])
