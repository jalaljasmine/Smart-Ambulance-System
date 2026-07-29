# pyrefly: ignore [missing-import]
import joblib

model = joblib.load("models/traffic_model.pkl")

def predict_traffic(input_data):
    prediction = model.predict(input_data)
    return prediction[0]
