import os
import joblib
import pandas as pd

# Load model and metadata
model_dir = os.path.dirname(__file__)
model_path = os.path.join(model_dir, "accident_model.pkl")
metadata_path = os.path.join(model_dir, "accident_model_metadata.pkl")

model = joblib.load(model_path)
metadata = joblib.load(metadata_path)

encoders = metadata["encoders"]
column_modes = metadata["column_modes"]
feature_cols = metadata["feature_cols"]

def predict(input_dict):
    """
    Predicts accident severity based on input features (6 core features).
    If input_dict is None, returns a default prediction "Serious Injury".
    """
    if not input_dict:
        return "Serious Injury"
        
    encoded_values = []
    for col in feature_cols:
        val = input_dict.get(col, column_modes[col])
        if col in encoders:
            le = encoders[col]
            val_str = str(val)
            # If the category is unseen, fall back to the mode
            if val_str not in le.classes_:
                mode_str = str(column_modes[col])
                if mode_str in le.classes_:
                    encoded_val = le.transform([mode_str])[0]
                else:
                    encoded_val = 0
            else:
                encoded_val = le.transform([val_str])[0]
        else:
            try:
                encoded_val = float(val)
                if pd.isna(encoded_val):
                    encoded_val = float(column_modes[col])
            except Exception:
                encoded_val = float(column_modes[col])
        encoded_values.append(encoded_val)
        
    # Convert to DataFrame with proper column names
    row_df = pd.DataFrame([encoded_values], columns=feature_cols)
                
    # Predict
    pred = model.predict(row_df)
    pred_val = int(pred[0])
    
    # Map encoded output back to string labels
    target_mapping = {
        0: "Fatal Injury",
        1: "Serious Injury",
        2: "Slight Injury"
    }
    return target_mapping.get(pred_val, "Serious Injury")