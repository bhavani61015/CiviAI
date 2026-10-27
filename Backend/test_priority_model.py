
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "priority_model_v2.joblib"

saved = joblib.load(MODEL_PATH)
model = saved["model"]
label_to_id = saved["label_to_id"]
id_to_label = {value: key for key, value in label_to_id.items()}

complaint = pd.DataFrame([{
    "category": "Pothole",
    "description": "A deep pothole on a busy road makes vehicles swerve",
    "detections_count": 3,
    "near_school_or_hospital": 0,
    "blocks_traffic": 1,
    "safety_risk": 1,
}])

predicted_id = int(model.predict(complaint)[0])
probabilities = model.predict_proba(complaint)[0]

print("Test complaint:", complaint.iloc[0]["description"])
print("Predicted priority:", id_to_label[predicted_id])
print(f"Model score: {probabilities[predicted_id]:.2%}")
print("\nDemo only: trained on synthetic sample data.")
