import os
import joblib

# Resolve path for traffic_model.pkl relative to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
model_path = os.path.join(project_dir, "models", "traffic_model.pkl")

model = joblib.load(model_path)

def predict_traffic(input_data):
    prediction = model.predict(input_data)
    return prediction[0]
