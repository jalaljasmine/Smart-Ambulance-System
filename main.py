"""
main.py -- Smart Ambulance System
==================================
Console entry point for the Smart Ambulance Routing & Traffic Optimization System.

Full pipeline:
    1. Collect accident scenario inputs (or use default demo)
    2. Predict accident severity (RandomForest ML model)
    3. Recommend the best hospital based on severity score
    4. Predict current traffic volume (RandomForest ML model)
    5. Calculate shortest route via Dijkstra's algorithm
    6. Display final results

Run from project root:
    python main.py
    python main.py --demo     (skip prompts, use preset accident scenario)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import datetime

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError with special chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from models.predict_severity import predict
from models.hospital_score    import recommend_hospital
from routing.check_traffic    import predict_traffic
from routing.dijkstra         import calculate_route
from config import DEFAULT_TRAFFIC_INPUT, FORM_OPTIONS


<<<<<<< HEAD
severity = predict(None)
print("\nPredicted Severity:", severity)
hospital = recommend_hospital(severity)
input_data = [[9, 2, 7, 1, 3]]
traffic = predict_traffic(input_data)

print("\nPredicted Traffic:", traffic)
route, distance, time = calculate_route(traffic)
=======
# ── Helpers ───────────────────────────────────────────────────────────────────

def _separator(char="=", width=50):
    print(char * width)


def _print_banner():
    _separator()
    print("   [*] SMART AMBULANCE ROUTING SYSTEM")
    print("       Vijayawada, Andhra Pradesh")
    _separator()
    print()

>>>>>>> 921f84e (Updated Smart Ambulance Routing Project)

def _get_demo_inputs() -> dict:
    """Return a realistic pre-set accident scenario for demo/testing."""
    return {
        "Day_of_week":             "Friday",
        "Weather_conditions":      "Raining",
        "Light_conditions":        "Darkness - lights lit",
        "Road_surface_conditions": "Wet or damp",
        "Cause_of_accident":       "Driving at high speed",
        "Type_of_collision":       "Vehicle with vehicle collision",
        "Number_of_vehicles_involved": 3,
        "Number_of_casualties":    4,
    }


def _get_user_inputs() -> dict:
    """Interactively collect accident details from the user."""
    print("📋 Enter Accident Details")
    _separator("-")

    inputs = {}
    options = FORM_OPTIONS

    # Day of Week
    days = options["Day_of_week"]
    print("Day of Week:")
    for i, d in enumerate(days, 1):
        print(f"  {i}. {d}")
    choice = input("Select (1-7) [default: 1=Monday]: ").strip() or "1"
    inputs["Day_of_week"] = days[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= 7 else "Monday"

    # Weather
    weathers = options["Weather_conditions"]
    print("\nWeather Conditions:")
    for i, w in enumerate(weathers, 1):
        print(f"  {i}. {w}")
    choice = input(f"Select (1-{len(weathers)}) [default: 1=Normal]: ").strip() or "1"
    inputs["Weather_conditions"] = weathers[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(weathers) else "Normal"

    # Light Conditions
    lights = options["Light_conditions"]
    print("\nLight Conditions:")
    for i, l in enumerate(lights, 1):
        print(f"  {i}. {l}")
    choice = input(f"Select (1-{len(lights)}) [default: 1=Daylight]: ").strip() or "1"
    inputs["Light_conditions"] = lights[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(lights) else "Daylight"

    # Road Surface
    surfaces = options["Road_surface_conditions"]
    print("\nRoad Surface Conditions:")
    for i, s in enumerate(surfaces, 1):
        print(f"  {i}. {s}")
    choice = input(f"Select (1-{len(surfaces)}) [default: 1=Dry]: ").strip() or "1"
    inputs["Road_surface_conditions"] = surfaces[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(surfaces) else "Dry"

    # Cause
    causes = options["Cause_of_accident"]
    print("\nCause of Accident:")
    for i, c in enumerate(causes, 1):
        print(f"  {i}. {c}")
    choice = input(f"Select (1-{len(causes)}) [default: 1=Overtaking]: ").strip() or "1"
    inputs["Cause_of_accident"] = causes[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(causes) else "Overtaking"

    # Number of vehicles
    choice = input("\nNumber of vehicles involved (1-7) [default: 2]: ").strip() or "2"
    inputs["Number_of_vehicles_involved"] = int(choice) if choice.isdigit() and 1 <= int(choice) <= 7 else 2

    # Number of casualties
    choice = input("Number of casualties (1-8) [default: 2]: ").strip() or "2"
    inputs["Number_of_casualties"] = int(choice) if choice.isdigit() and 1 <= int(choice) <= 8 else 2

    return inputs


def _build_traffic_input() -> list:
    """Build traffic model feature vector from current datetime."""
    now = datetime.datetime.now()
    return [[now.hour, now.day, now.month, now.weekday(), 3]]


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(accident_inputs: dict = None) -> dict:
    """
    Execute the complete Smart Ambulance pipeline.

    Args:
        accident_inputs (dict): Accident feature values. Uses demo defaults if None.

    Returns:
        dict: Results containing severity, hospital, traffic, distance, and ETA.
    """
    if accident_inputs is None:
        accident_inputs = _get_demo_inputs()

    results = {}

    # -- Step 1: Accident Severity Prediction --------------------------------
    print("\n[1] Predicting Accident Severity...")
    severity = predict(accident_inputs)
    results["severity"] = severity
    print(f"    Severity: {severity}")

    # -- Step 2: Hospital Recommendation ------------------------------------
    print("\n[2] Recommending Best Hospital...")
    hospital = recommend_hospital(severity)
    results["hospital"]      = hospital["Hospital"]
    results["hospital_lat"]  = float(hospital["Latitude"])
    results["hospital_lon"]  = float(hospital["Longitude"])
    results["icu_beds"]      = int(hospital["ICU_Beds"])
    results["trauma_center"] = hospital["Trauma_Center"]
    print(f"    Hospital    : {hospital['Hospital']}")
    print(f"    ICU Beds    : {hospital['ICU_Beds']}  | Trauma Center: {hospital['Trauma_Center']}")

    # -- Step 3: Traffic Prediction -----------------------------------------
    print("\n[3] Predicting Traffic Volume...")
    traffic_input = _build_traffic_input()
    traffic = predict_traffic(traffic_input)
    results["traffic"] = round(traffic, 2)
    print(f"    Predicted Vehicles/Hour: {traffic:.2f}")

    # -- Step 4: Shortest Route via Dijkstra --------------------------------
    print("\n[4] Calculating Shortest Route (Dijkstra)...")
    route, route_coords, distance_m, eta = calculate_route(
        traffic    = traffic,
        dest_lat   = results["hospital_lat"],
        dest_lon   = results["hospital_lon"],
    )
    results["route_nodes"]  = len(route)
    results["distance_km"]  = round(distance_m / 1000, 2)
    results["eta_minutes"]  = round(eta, 2)
    results["route_coords"] = route_coords

    return results


def main():
    _print_banner()

    # Check for --demo flag
    demo_mode = "--demo" in sys.argv

    if demo_mode:
        print("[DEMO] Running with preset accident scenario\n")
        inputs = _get_demo_inputs()
        print("Demo Scenario:")
        for k, v in inputs.items():
            print(f"  {k}: {v}")
    else:
        inputs = _get_user_inputs()

    # Run the full pipeline
    results = run_pipeline(inputs)

    # -- Final Results Display -----------------------------------------------
    print()
    _separator("=")
    print("             FINAL RESULTS")
    _separator("=")
    print(f"  Accident Severity   : {results['severity']}")
    print(f"  Recommended Hospital: {results['hospital']}")
    print(f"  ICU Beds            : {results['icu_beds']}")
    print(f"  Trauma Center       : {results['trauma_center']}")
    print(f"  Traffic (veh/hr)    : {results['traffic']}")
    print(f"  Route Distance      : {results['distance_km']} km")
    print(f"  Estimated ETA       : {results['eta_minutes']} minutes")
    print(f"  Route Nodes         : {results['route_nodes']}")
    _separator("=")
    print()
    print("[OK] Pipeline completed successfully.")
    print("[TIP] Run `python dashboard/app.py` to launch the web dashboard.")


if __name__ == "__main__":
    main()