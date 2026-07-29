"""
models/predict_severity.py — Smart Ambulance System
====================================================
Loads the pre-trained RandomForestClassifier and provides
the predict() function for accident severity classification.

Model was trained on 31 features from accidents.csv.
All categorical columns were LabelEncoded alphabetically.
This module replicates that exact encoding at inference time.

Classes predicted:
    0 → Fatal Injury
    1 → Serious Injury
    2 → Slight Injury
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys

# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Allow import from project root regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ACCIDENT_MODEL_PATH, ACCIDENTS_CSV, SEVERITY_LABELS

# ── Module-level state (loaded once on first call) ────────────────────────────
_model          = None          # Trained RandomForestClassifier
_encoders       = {}            # {col_name: fitted LabelEncoder}
_feature_cols   = []            # Ordered list of feature column names
_raw_defaults   = {}            # Default raw values for each feature column


def _initialize() -> None:
    """
    Load the trained model and build LabelEncoders by fitting them
    on the same training data — exactly replicating the training process.

    This is called lazily on the first call to predict().
    """
    global _model, _encoders, _feature_cols, _raw_defaults

    # 1. Load the saved model
    _model = joblib.load(ACCIDENT_MODEL_PATH)

    # 2. Load and preprocess training data (same steps as training notebook)
    df = pd.read_csv(ACCIDENTS_CSV)
    df = df.drop_duplicates()

    # Fill NaN — categorical/str → mode, numeric → median
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Separate features from target
    X = df.drop("Accident_severity", axis=1)
    _feature_cols = X.columns.tolist()

    # 4. Store raw (pre-encoding) mode/median defaults for every column
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            _raw_defaults[col] = float(X[col].median())
        else:
            _raw_defaults[col] = str(X[col].mode()[0])

    # 5. Fit a LabelEncoder per non-numeric column
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            enc = LabelEncoder()
            enc.fit(X[col].astype(str))
            _encoders[col] = enc


def predict(input_dict: dict = None) -> str:
    """
    Predict accident severity from input features.

    Args:
        input_dict (dict, optional): Feature key-value pairs.
            Missing keys are filled with dataset mode/median defaults.
            Pass None for a default demo prediction.

    Returns:
        str: One of "Slight Injury", "Serious Injury", "Fatal Injury"

    Example:
        >>> predict({"Day_of_week": "Monday", "Weather_conditions": "Raining"})
        "Serious Injury"
    """
    global _model

    # Lazy initialization — only loads data on first call
    if _model is None:
        print("  [Severity Model] Loading model and building encoders...")
        _initialize()
        print("  [Severity Model] Ready.")

    if input_dict is None:
        input_dict = {}

    # Build feature row: start with defaults, then override with user input
    row = dict(_raw_defaults)
    row.update({k: str(v) if k in _encoders else v for k, v in input_dict.items()})

    # Encode and assemble feature vector in the trained column order
    encoded = []
    for col in _feature_cols:
        val = row.get(col, _raw_defaults[col])
        if col in _encoders:
            try:
                encoded_val = int(_encoders[col].transform([str(val)])[0])
            except ValueError:
                # Unseen label — fallback to 0 (alphabetically first class)
                encoded_val = 0
        else:
            try:
                encoded_val = float(val)
            except (ValueError, TypeError):
                encoded_val = 0.0
        encoded.append(encoded_val)

    # Run prediction
    prediction = int(_model.predict([encoded])[0])
    return SEVERITY_LABELS.get(prediction, "Serious Injury")


def get_unique_values(column: str) -> list:
    """
    Return sorted unique values for a given feature column.
    Useful for populating form dropdowns.

    Args:
        column (str): Column name from accidents.csv features.

    Returns:
        list: Sorted unique string values.
    """
    if not _encoders:
        _initialize()
    if column in _encoders:
        return list(_encoders[column].classes_)
    return []


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing predict_severity.py...")

    # Test 1: Default prediction (mode values)
    result = predict()
    print(f"Default prediction: {result}")

    # Test 2: Custom input — rainy night, high speed
    custom = {
        "Day_of_week":            "Friday",
        "Weather_conditions":     "Raining",
        "Light_conditions":       "Darkness - lights lit",
        "Road_surface_conditions": "Wet or damp",
        "Cause_of_accident":      "Driving at high speed",
        "Number_of_vehicles_involved": 3,
        "Number_of_casualties":   4,
    }
    result2 = predict(custom)
    print(f"High-risk scenario: {result2}")