"""
models/hospital_score.py — Smart Ambulance System
==================================================
Scores and ranks hospitals from hospitals.csv based on accident severity.

Scoring logic:
    Slight Injury  → Trauma Center: +10 pts | ICU: ×0.2 | Staff: ×0.2
    Serious Injury → Trauma Center: +40 pts | ICU: ×0.3 | Staff: ×0.3
    Fatal Injury   → Trauma Center: +60 pts | ICU: ×0.4 | Staff: ×0.4

The hospital with the highest total score is recommended.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# Allow import from project root regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HOSPITALS_CSV


import math
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

# Allow import from project root regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HOSPITALS_CSV, DEFAULT_ACCIDENT_LAT, DEFAULT_ACCIDENT_LON


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in kilometres between two coordinate pairs using Haversine formula."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def recommend_hospital(
    severity: str,
    strategy: str = "capacity",
    selected_hospital: str = None,
    start_lat: float = DEFAULT_ACCIDENT_LAT,
    start_lon: float = DEFAULT_ACCIDENT_LON,
) -> pd.Series:
    """
    Score all hospitals and return the best one based on severity, strategy, or manual selection.

    Args:
        severity (str): One of "Slight Injury", "Serious Injury", "Fatal Injury".
        strategy (str): "capacity" (default), "proximity" (nearest distance heavy), or "manual".
        selected_hospital (str): Specific hospital name if manually chosen.
        start_lat (float): Accident latitude.
        start_lon (float): Accident longitude.

    Returns:
        pd.Series: Row from hospitals.csv with an added "Score" field.
    """
    ranked = get_all_hospitals_ranked(
        severity=severity,
        strategy=strategy,
        selected_hospital=selected_hospital,
        start_lat=start_lat,
        start_lon=start_lon,
    )
    return ranked.iloc[0]


def get_all_hospitals_ranked(
    severity: str,
    strategy: str = "capacity",
    selected_hospital: str = None,
    start_lat: float = DEFAULT_ACCIDENT_LAT,
    start_lon: float = DEFAULT_ACCIDENT_LON,
) -> pd.DataFrame:
    """
    Return all hospitals ranked by score according to the chosen routing strategy.

    Strategies:
      - 'capacity' : Standard capacity & trauma score (default)
      - 'proximity': Proximity-weighted score (closer hospitals get high bonus)
      - 'manual'   : Places the user's selected hospital at rank #1
    """
    hospitals = pd.read_csv(HOSPITALS_CSV)

    # Calculate distance to each hospital first
    distances = []
    for _, row in hospitals.iterrows():
        dist = _haversine_distance(start_lat, start_lon, float(row["Latitude"]), float(row["Longitude"]))
        distances.append(round(dist, 2))

    hospitals = hospitals.copy()
    hospitals["Distance_km"] = distances

    # Handle manual explicit hospital override
    if selected_hospital:
        # Check case-insensitive match or exact match
        matched = hospitals[hospitals["Hospital"].str.strip().str.lower() == selected_hospital.strip().lower()]
        if not matched.empty:
            override_name = matched.iloc[0]["Hospital"]
            # Give selected hospital top score boost so it ranks #1
            scores = []
            for _, row in hospitals.iterrows():
                if row["Hospital"] == override_name:
                    scores.append(999.0)
                else:
                    scores.append(10.0)
            hospitals["Score"] = scores
            return hospitals.sort_values(by="Score", ascending=False).reset_index(drop=True)

    scores = []
    for _, row in hospitals.iterrows():
        s = 0.0
        has_trauma = (row["Trauma_Center"] == "Yes")
        dist = row["Distance_km"]  # straight-line km from accident

        # ── Proximity Score (PRIMARY — 70% weight) ─────────────────────────
        # Max realistic city distance = 15 km → score from 0‒70
        # Closer hospital scores higher.  Distance > 15 km → 0 proximity pts.
        proximity_score = max(0.0, 70.0 - (dist * (70.0 / 15.0)))

        # ── Capacity Score (SECONDARY — 30% weight) ──────────────────────────
        capacity_score = 0.0
        if severity == "Slight Injury":
            if has_trauma: capacity_score += 5
            capacity_score += row["ICU_Beds"] * 0.08 + row["Emergency_Staff"] * 0.07
        elif severity == "Serious Injury":
            if has_trauma: capacity_score += 15
            capacity_score += row["ICU_Beds"] * 0.10 + row["Emergency_Staff"] * 0.09
        else:  # Fatal Injury
            if has_trauma: capacity_score += 25
            capacity_score += row["ICU_Beds"] * 0.12 + row["Emergency_Staff"] * 0.10

        # Clamp capacity score to max 30 pts so distance always dominates
        capacity_score = min(30.0, capacity_score)

        if strategy == "proximity":
            # Pure proximity: distance is everything
            s = proximity_score + (capacity_score * 0.1)
        else:
            # Default "capacity" strategy still respects distance heavily
            s = proximity_score * 0.7 + capacity_score

        scores.append(round(s, 2))

    hospitals["Score"] = scores
    return hospitals.sort_values(by="Score", ascending=False).reset_index(drop=True)



# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from models.predict_severity import predict

    severity = predict()
    print(f"Predicted Severity: {severity}")

    print("\n--- Capacity Strategy (Default) ---")
    best_cap = recommend_hospital(severity, strategy="capacity")
    print(f"Best: {best_cap['Hospital']} | Score: {best_cap['Score']}")

    print("\n--- Proximity Strategy (Nearest Hospital) ---")
    best_prox = recommend_hospital(severity, strategy="proximity", start_lat=16.5200, start_lon=80.6300)
    print(f"Best: {best_prox['Hospital']} | Score: {best_prox['Score']} | Dist: {best_prox['Distance_km']}km")

    print("\n--- Manual Override (e.g. Manipal Hospital Vijayawada) ---")
    best_man = recommend_hospital(severity, selected_hospital="Manipal Hospital Vijayawada")
    print(f"Best: {best_man['Hospital']} | Score: {best_man['Score']}")