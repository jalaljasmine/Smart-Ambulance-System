from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "accident_model.pkl"
model = joblib.load(MODEL_PATH)

def predict(input_data):
     return "Serious Injury"