"""
config.py — Smart Ambulance System
====================================
Central configuration file for all paths, constants, and model settings.
All other modules import from here — no more hardcoded paths.
"""

import os

# ============================================================
# Project Root (absolute path, works from any working directory)
# ============================================================
ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Dataset Paths
# ============================================================
ACCIDENTS_CSV = os.path.join(ROOT, "datasets", "accidents.csv")
HOSPITALS_CSV = os.path.join(ROOT, "datasets", "hospitals.csv")
TRAFFIC_CSV   = os.path.join(ROOT, "datasets", "traffic.csv")
GRAPHML_PATH  = os.path.join(ROOT, "datasets", "vijayawada.graphml")

# ============================================================
# Model Paths
# ============================================================
ACCIDENT_MODEL_PATH = os.path.join(ROOT, "models", "accident_model.pkl")    # Predict Accident Severity
TRAFFIC_MODEL_PATH  = os.path.join(ROOT, "models", "traffic_model.pkl")     # Predict Traffic Volume

# ============================================================
# Severity Label Map
# LabelEncoder sorts classes alphabetically:
#   0 → "Fatal Injury", 1 → "Serious Injury", 2 → "Slight Injury"
# ============================================================
SEVERITY_LABELS = {
    0: "Fatal Injury",
    1: "Serious Injury",
    2: "Slight Injury",
}

# ============================================================
# Traffic → Speed Mapping (predicted vehicles/hour → km/h)
# ============================================================
def get_speed_from_traffic(vehicle_count: float) -> int:
    """Return ambulance speed (km/h) based on predicted traffic volume."""
    if vehicle_count < 10:
        return 40   # light traffic
    elif vehicle_count < 20:
        return 30   # moderate traffic
    else:
        return 20   # heavy traffic

# ============================================================
# Default Demo Accident Scenario Coordinates
# Benz Circle / Auto Nagar area, Vijayawada (real road location)
# ============================================================
DEFAULT_ACCIDENT_LAT = 16.4985
DEFAULT_ACCIDENT_LON = 80.6580

# ============================================================
# Traffic Prediction Default Input
# Features: [Hour, Day, Month, Weekday, Junction]
# Example: 9 AM, 2nd day, July, Tuesday (1), Junction 3
# ============================================================
DEFAULT_TRAFFIC_INPUT = [[9, 2, 7, 1, 3]]

# ============================================================
# Vijayawada Pre-Set Landmarks for Easy Location Selection
# ============================================================
VIJAYAWADA_LANDMARKS = [
    {"name": "Benz Circle (MG Road)", "lat": 16.5022, "lon": 80.6475},
    {"name": "Auto Nagar (Bandar Road)", "lat": 16.4985, "lon": 80.6580},
    {"name": "Vijayawada Railway Station", "lat": 16.5181, "lon": 80.6203},
    {"name": "Pandit Nehru Bus Station (PNBS)", "lat": 16.5060, "lon": 80.6180},
    {"name": "Governorpet (Eluru Road)", "lat": 16.5125, "lon": 80.6280},
    {"name": "Gunadala Centre", "lat": 16.5250, "lon": 80.6620},
    {"name": "Ramavarappadu Ring", "lat": 16.5230, "lon": 80.6720},
    {"name": "Bhavanipuram", "lat": 16.5280, "lon": 80.5950},
    {"name": "Kanuru Junction", "lat": 16.4850, "lon": 80.6800},
    {"name": "NTR Health University / Siddhartha College", "lat": 16.5010, "lon": 80.6520},
]

# Load list of hospitals for selection dropdowns
try:
    import pandas as pd
    _hosp_df = pd.read_csv(HOSPITALS_CSV)
    ALL_HOSPITALS = _hosp_df["Hospital"].tolist()
except Exception:
    ALL_HOSPITALS = [
        "Government General Hospital",
        "Manipal Hospital Vijayawada",
        "NRI Hospital",
        "Care Hospital Vijayawada",
        "Ramesh Hospital",
        "Andhra Hospital",
        "Vijaya Super Speciality Hospital",
        "Sentini Hospital",
        "Sai Super Speciality Hospital",
        "Praveen Hospital"
    ]

# ============================================================
# Form Dropdown Options (for Flask dashboard)
# These match the exact values in the training dataset.
# ============================================================
FORM_OPTIONS = {
    "Landmarks": VIJAYAWADA_LANDMARKS,
    "Hospitals": ALL_HOSPITALS,
    "Day_of_week": [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ],
    "Road_surface_conditions": [
        "Dry", "Wet or damp", "Snow", "Flood over 3cm. deep"
    ],
    "Weather_conditions": [
        "Normal", "Raining", "Raining and Windy", "Windy",
        "Cloudy", "Snow", "Fog or mist", "Unknown"
    ],
    "Light_conditions": [
        "Daylight",
        "Darkness - lights lit",
        "Darkness - no lighting",
        "Darkness - lights unlit"
    ],
    "Type_of_collision": [
        "Vehicle with vehicle collision",
        "Collision with roadside objects",
        "Collision with roadside-parked vehicles",
        "Rollover",
        "Fall from vehicle",
        "Collision with animals",
        "Unknown",
        "With Train"
    ],
    "Cause_of_accident": [
        "Overtaking",
        "Changing lane to the left",
        "Changing lane to the right",
        "Driving at high speed",
        "Driving carelessly",
        "No distancing",
        "Moving Backward",
        "Turnover",
        "Other",
        "Overloading",
        "Unknown"
    ],
    "Number_of_vehicles_involved": list(range(1, 8)),
    "Number_of_casualties":        list(range(1, 9)),
    "Junction": list(range(1, 5)),
}

