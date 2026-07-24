import os
import math
import pandas as pd

def haversine_distance(coord1, coord2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def get_sorted_hospitals(severity: str, start_coords=None, scoring_mode: str = "original"):
    """
    Ranks hospitals by clinical capability and proximity.

    Args:
        severity (str): One of "Slight Injury", "Serious Injury", "Fatal Injury".
        start_coords (tuple | None): (lat, lon) of the accident site.
        scoring_mode (str): ``"original"`` uses the raw ICU_Beds / distance formula.
            ``"normalized"`` min-max-normalises both dimensions to 0-100 before
            weighting so bed count does not structurally dominate across the city.

    Returns:
        pd.DataFrame: hospitals sorted by Score descending, Score column attached.
    """
    if scoring_mode == "normalized":
        return get_sorted_hospitals_normalized(severity, start_coords)

    # Resolve hospitals.csv path dynamically to support calls from different directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    csv_path = os.path.join(project_dir, "datasets", "hospitals.csv")
    
    hospitals = pd.read_csv(csv_path)
    scores = []

    for index, row in hospitals.iterrows():
        s = 0
        
        # 1. Base Score on Severity & Resource Capability
        if severity == "Slight Injury":
            if row["Trauma_Center"] == "Yes":
                s += 10
            s += row["ICU_Beds"] * 0.2
            s += row["Emergency_Staff"] * 0.2
            
            # Apply proximity penalty if coordinates are provided
            if start_coords:
                dist = haversine_distance(start_coords, (row["Latitude"], row["Longitude"]))
                s -= dist * 3.0  # 3 points penalty per km for slight injuries
                
        elif severity == "Serious Injury":
            if row["Trauma_Center"] == "Yes":
                s += 40
            s += row["ICU_Beds"] * 0.3
            s += row["Emergency_Staff"] * 0.3
            
            # Apply proximity penalty if coordinates are provided
            if start_coords:
                dist = haversine_distance(start_coords, (row["Latitude"], row["Longitude"]))
                s -= dist * 5.0  # 5 points penalty per km for serious injuries
                
        elif severity == "Fatal Injury":
            if row["Trauma_Center"] == "Yes":
                s += 60
            s += row["ICU_Beds"] * 0.4
            s += row["Emergency_Staff"] * 0.4
            
            # Apply proximity penalty if coordinates are provided
            if start_coords:
                dist = haversine_distance(start_coords, (row["Latitude"], row["Longitude"]))
                s -= dist * 8.0  # 8 points penalty per km for fatal injuries (time is extremely critical)

        scores.append(s)

    hospitals["Score"] = scores
    hospitals = hospitals.sort_values(by="Score", ascending=False).reset_index(drop=True)
    return hospitals


def get_sorted_hospitals_normalized(severity: str, start_coords=None):
    """
    Normalized variant of the hospital scoring function.

    Both ICU_Beds (capacity signal) and haversine distance (proximity signal) are
    min-max-normalised to [0, 100] across the current candidate set before weights
    are applied.  This prevents the raw bed count from structurally dominating the
    ranking regardless of city scale, enabling all 10 hospitals to appear across
    varied call locations.

    Weights by severity:
        - Slight  : 35% capacity + 65% proximity
        - Serious : 30% capacity + 70% proximity
        - Fatal   : 25% capacity + 75% proximity
    Trauma-center bonus: 20 / 40 / 60 points (Slight / Serious / Fatal).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    csv_path = os.path.join(project_dir, "datasets", "hospitals.csv")

    hospitals = pd.read_csv(csv_path)

    # --- Compute raw values ---
    beds = hospitals["ICU_Beds"].astype(float)
    staff = hospitals["Emergency_Staff"].astype(float)

    # --- Min-max normalize beds and staff to 0-100 ---
    beds_range = beds.max() - beds.min()
    norm_beds = (beds - beds.min()) / beds_range * 100 if beds_range > 0 else pd.Series([50.0] * len(hospitals))

    staff_range = staff.max() - staff.min()
    norm_staff = (staff - staff.min()) / staff_range * 100 if staff_range > 0 else pd.Series([50.0] * len(hospitals))

    # --- Compute distances and normalize proximity (closer = higher score) ---
    if start_coords:
        dists = hospitals.apply(
            lambda r: haversine_distance(start_coords, (r["Latitude"], r["Longitude"])), axis=1
        )
        dist_range = dists.max() - dists.min()
        # Invert: closest hospital gets 100, farthest gets 0
        norm_proximity = (1 - (dists - dists.min()) / dist_range) * 100 if dist_range > 0 else pd.Series([50.0] * len(hospitals))
    else:
        norm_proximity = pd.Series([50.0] * len(hospitals))

    # --- Compute score per severity ---
    severity_cfg = {
        "Slight Injury":  {"trauma_bonus": 20, "w_beds": 0.35, "w_staff": 0.35, "w_prox": 0.65},
        "Serious Injury": {"trauma_bonus": 40, "w_beds": 0.30, "w_staff": 0.30, "w_prox": 0.70},
        "Fatal Injury":   {"trauma_bonus": 60, "w_beds": 0.25, "w_staff": 0.25, "w_prox": 0.75},
    }
    cfg = severity_cfg.get(severity, severity_cfg["Serious Injury"])

    trauma_bonus = hospitals["Trauma_Center"].apply(lambda x: cfg["trauma_bonus"] if x == "Yes" else 0)
    scores = (
        trauma_bonus
        + cfg["w_beds"]  * norm_beds
        + cfg["w_staff"] * norm_staff
        + cfg["w_prox"]  * norm_proximity
    )

    hospitals["Score"] = scores
    hospitals = hospitals.sort_values(by="Score", ascending=False).reset_index(drop=True)
    return hospitals

def recommend_hospital(severity: str, start_coords=None, scoring_mode: str = "original"):
    """
    Recommends the best hospital based on clinical capability and proximity.

    Args:
        severity (str): Accident severity label.
        start_coords (tuple | None): (lat, lon) of the accident site.
        scoring_mode (str): ``"original"`` or ``"normalized"`` (see get_sorted_hospitals).
    """
    hospitals = get_sorted_hospitals(severity, start_coords, scoring_mode=scoring_mode)
    best = hospitals.iloc[0]
    return best

def recommend_hospital_with_lock(
        severity: str,
        lock_manager,
        ambulance_id: str,
        current_time: float,
        start_coords=None,
        traffic=None,
        buffer: float = 8.0,
        scoring_mode: str = "original",
):
    """
    Attempts to secure a reservation lock at the best eligible hospital.
    Iterates through candidate hospitals sorted by score.

    Args:
        severity (str): Accident severity label.
        lock_manager (LockManager): Active LockManager instance.
        ambulance_id (str): Unique identifier for the dispatched ambulance.
        current_time (float): Current simulation time in minutes.
        start_coords (tuple | None): (lat, lon) of the accident site.
        traffic (float | None): Traffic volume passed to Dijkstra for ETA estimate.
        buffer (float): Extra minutes added to expected_arrival to set lock TTL.
        scoring_mode (str): ``"original"`` or ``"normalized"``.

    Returns:
        tuple: (hospital_row, reservation_lock) or (top_scored_hospital_row, None)
               if no beds are available at any candidate.
    """
    if start_coords is None:
        start_coords = (16.5062, 80.6480)
        
    sorted_hospitals = get_sorted_hospitals(severity, start_coords, scoring_mode=scoring_mode)
    
    # Import routing inside to avoid circular dependencies
    from routing.dijkstra import calculate_route
    
    for _, row in sorted_hospitals.iterrows():
        hospital_name = row["Hospital"]
        dest_coords = (row["Latitude"], row["Longitude"])
        
        # Calculate ETA to this hospital
        try:
            _, _, travel_time_min = calculate_route(start_coords, dest_coords, traffic=traffic)
        except Exception:
            dist_km = haversine_distance(start_coords, dest_coords)
            travel_time_min = (dist_km / 40.0) * 60.0
            
        expected_arrival = current_time + travel_time_min
        
        # Attempt lock claim
        lock = lock_manager.claim_lock(hospital_name, ambulance_id, expected_arrival, current_time, buffer=buffer)
        if lock:
            return row, lock
            
    # Fallback to the top-scored hospital if no beds are available anywhere
    return sorted_hospitals.iloc[0], None

if __name__ == "__main__":
    from models.predict_severity import predict
    severity = predict(None)
    print("Predicted Severity:", severity)
    
    # Test without coordinates
    best = recommend_hospital(severity)
    print(f"\nBest Hospital (No Proximity Penalty): {best['Hospital']} (Score: {best['Score']})")
    
    # Test with coordinates near NRI Hospital
    nri_coords = (16.5415, 80.5150)
    best_nri = recommend_hospital(severity, start_coords=nri_coords)
    print(f"Best Hospital (Near NRI Hospital): {best_nri['Hospital']} (Score: {best_nri['Score']})")