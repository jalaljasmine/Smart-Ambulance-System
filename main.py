import warnings
import os
warnings.filterwarnings("ignore")

# Load dotenv if present (to read TOMTOM_API_KEY from .env file)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models.predict_severity import predict
from models.hospital_score import recommend_hospital_with_lock
from routing.check_traffic import predict_traffic
from routing.dijkstra import calculate_route
from coordination.lock_manager import LockManager

def prompt_menu(title, options):
    print(f"\n--- {title} ---")
    for idx, opt in enumerate(options, 1):
        print(f"{idx}) {opt}")
    while True:
        try:
            choice = int(input("Select option: "))
            if 1 <= choice <= len(options):
                return choice
            print(f"Invalid option. Please choose between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number.")

def get_ml_features():
    print("\n--- Provide Accident Features (Press Enter to use typical values) ---")
    
    day = input("Day of week (Monday, Tuesday, etc.) [default: Friday]: ").strip()
    if not day: day = "Friday"
    
    age = input("Age band of driver (18-30, 31-50, Over 51, Under 18) [default: 18-30]: ").strip()
    if not age: age = "18-30"
    
    light = input("Light conditions (Daylight, Darkness - lights lit, etc.) [default: Daylight]: ").strip()
    if not light: light = "Daylight"
    
    weather = input("Weather conditions (Normal, Raining, etc.) [default: Normal]: ").strip()
    if not weather: weather = "Normal"
        
    try:
        vehicles = input("Number of vehicles involved [default: 2]: ").strip()
        vehicles = int(vehicles) if vehicles else 2
    except ValueError:
        vehicles = 2
        
    try:
        casualties = input("Number of casualties [default: 1]: ").strip()
        casualties = int(casualties) if casualties else 1
    except ValueError:
        casualties = 1
        
    return {
        "Day_of_week": day,
        "Age_band_of_driver": age,
        "Light_conditions": light,
        "Weather_conditions": weather,
        "Number_of_vehicles_involved": vehicles,
        "Number_of_casualties": casualties
    }

def main():
    print("==================================================")
    print("====== SMART AMBULANCE ROUTING & TRAFFIC SYSTEM ======")
    print("=========== Reservation-Lock Edition ==============")
    print("==================================================")
    
    # Initialise LockManager (in-memory, swappable for Redis/SQLite)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "datasets", "hospitals.csv")
    lock_manager = LockManager(csv_path)
    print("\nLockManager initialised — bed pools loaded from hospitals.csv")
    for h_name, pool in lock_manager.pools.items():
        print(f"  {h_name}: {pool.total_beds} ICU beds")
    
    # 1. Accident Severity Mode selection
    mode = prompt_menu(
        "Accident Severity Assessment Mode", 
        ["Manual Override (Direct Input)", "Machine Learning Model Prediction"]
    )
    
    if mode == 1:
        # Manual Input
        severity_choice = prompt_menu(
            "Select Accident Severity Level",
            ["Slight Injury", "Serious Injury", "Fatal Injury"]
        )
        severities = ["Slight Injury", "Serious Injury", "Fatal Injury"]
        severity = severities[severity_choice - 1]
    else:
        # ML Prediction Mode
        input_features = get_ml_features()
        print("\nRunning Machine Learning classification model...")
        severity = predict(input_features)
        
    print(f"\n>>> Determined Accident Severity: {severity}")
    
    # 2. Coordinate input
    print("\n--- Accident Coordinates ---")
    coords_input = input("Enter Accident Coordinates (latitude, longitude) [default: 16.5062, 80.6480]: ").strip()
    if coords_input:
        try:
            lat, lon = map(float, coords_input.split(","))
            start_coords = (lat, lon)
        except Exception:
            print("Invalid format. Falling back to default coordinates (16.5062, 80.6480).")
            start_coords = (16.5062, 80.6480)
    else:
        start_coords = (16.5062, 80.6480)
        
    # 3. Recommended Hospital with Reservation Lock
    import datetime
    now = datetime.datetime.now()
    ambulance_id = f"main_amb_{now.strftime('%H%M%S')}"
    
    hospital, lock = recommend_hospital_with_lock(
        severity=severity,
        lock_manager=lock_manager,
        ambulance_id=ambulance_id,
        current_time=now.timestamp() / 60.0,
        start_coords=start_coords
    )
    
    print(f"\n>>> Recommended Hospital based on severity & proximity details:")
    print(f"    Name            : {hospital['Hospital']}")
    print(f"    Trauma Center   : {hospital['Trauma_Center']}")
    print(f"    ICU Beds Total  : {lock_manager.pools[hospital['Hospital']].total_beds}")
    print(f"    Emergency Staff : {hospital['Emergency_Staff']}")
    print(f"    Coordinates     : {hospital['Latitude']}, {hospital['Longitude']}")
    
    if lock:
        pool = lock_manager.pools[hospital['Hospital']]
        print(f"    Lock Status     : BED PRE-RESERVED (lock_id={lock.lock_id})")
        print(f"    Beds Available  : {pool.available_beds} (after reservation)")
    else:
        print(f"    Lock Status     : WARNING — No beds available, hospital selected as best fallback")
    
    dest_coords = (hospital["Latitude"], hospital["Longitude"])
    
    # 4. Traffic Prediction & Routing
    # Call offline ML traffic prediction model first as context info
    import datetime
    now = datetime.datetime.now()
    
    print("\n--- Traffic Forecast Settings ---")
    use_current = input("Use current system time & default junction (Junction 3) for traffic forecast? (y/n) [default: y]: ").strip().lower()
    if use_current == 'n':
        try:
            hour = int(input("Enter Hour (0-23) [default: 9]: ") or 9)
            day = int(input("Enter Day of Month (1-31) [default: 2]: ") or 2)
            month = int(input("Enter Month (1-12) [default: 7]: ") or 7)
            weekday = int(input("Enter Weekday (0=Mon, 6=Sun) [default: 1]: ") or 1)
            junction = int(input("Enter Junction ID (1-4) [default: 3]: ") or 3)
        except ValueError:
            print("Invalid input. Using current system time and default junction.")
            hour, day, month, weekday, junction = now.hour, now.day, now.month, now.weekday(), 3
    else:
        hour, day, month, weekday, junction = now.hour, now.day, now.month, now.weekday(), 3
        
    traffic_input = [[hour, day, month, weekday, junction]]
    traffic_prediction = predict_traffic(traffic_input)
    print(f"\nLocal ML Model Predicted Traffic Volume (Vehicles/hr): {round(traffic_prediction, 1)}")
    
    # Run Routing
    print("\nCalculating optimal route to hospital...")
    try:
        route, distance_m, time_min = calculate_route(start_coords, dest_coords, traffic=traffic_prediction)
        
        # Display route results
        print("\n=================== ROUTING RESULT ===================")
        print(f"Recommended Hospital: {hospital['Hospital']}")
        print(f"Hospital Coordinates: {dest_coords}")
        print(f"Accident Coordinates: {start_coords}")
        print(f"Total Distance      : {round(distance_m / 1000, 2)} km")
        print(f"Estimated Time      : {round(time_min, 2)} minutes")
        print(f"Route Path Points   : {len(route)} nodes/points calculated")
        
        # Simulate ambulance arriving — convert reservation lock to occupied
        if lock:
            lock_manager.convert_to_occupied(lock.lock_id)
            pool = lock_manager.pools[hospital['Hospital']]
            print(f"\nLock converted to OCCUPIED — Bed secured at {hospital['Hospital']}")
            print(f"Remaining available beds at {hospital['Hospital']}: {pool.available_beds}")
        print("======================================================")
        
    except Exception as e:
        print(f"Error during route calculation: {e}")

if __name__ == "__main__":
    main()