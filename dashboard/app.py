"""
dashboard/app.py — Smart Ambulance System
==========================================
Flask web dashboard for the Smart Ambulance Routing System.

Routes:
    GET  /           → Input form (accident details)
    POST /predict    → Run full pipeline, display results
    GET  /map        → Full-screen Folium route map

Run from project root:
    python dashboard/app.py

Then open: http://localhost:5000
"""

import os
import sys
import warnings
import datetime
import json

warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, jsonify
import folium

# ── Path setup ────────────────────────────────────────────────────────────────
# This file lives in dashboard/; project root is one level up.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    FORM_OPTIONS,
    DEFAULT_ACCIDENT_LAT,
    DEFAULT_ACCIDENT_LON,
    DEFAULT_TRAFFIC_INPUT,
)
from models.predict_severity import predict
from models.hospital_score   import recommend_hospital, get_all_hospitals_ranked
from routing.check_traffic   import predict_traffic
from routing.dijkstra        import calculate_route

# ── Flask app setup ───────────────────────────────────────────────────────────
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates")
STATIC_DIR   = os.path.join(PROJECT_ROOT, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = "smart-ambulance-2024"

# ── Route map storage (in-memory for session) ─────────────────────────────────
_last_map_html = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_folium_map(
    start_lat, start_lon,
    dest_lat,  dest_lon,
    route_coords,
    hospital_name,
    distance_km,
    eta_minutes,
    severity,
) -> str:
    """Generate a Folium interactive map HTML string."""

    # Centre the map between start and destination
    centre_lat = (start_lat + dest_lat) / 2
    centre_lon = (start_lon + dest_lon) / 2

    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=14,
        tiles="CartoDB dark_matter",
    )

    # Accident marker (red)
    folium.Marker(
        location=[start_lat, start_lon],
        popup=folium.Popup(
            f"<b>🚨 Accident Location</b><br>Severity: {severity}",
            max_width=250,
        ),
        tooltip="Accident Location",
        icon=folium.Icon(color="red", icon="exclamation-sign"),
    ).add_to(m)

    # Hospital marker (green)
    folium.Marker(
        location=[dest_lat, dest_lon],
        popup=folium.Popup(
            f"<b>🏥 {hospital_name}</b><br>"
            f"Distance: {distance_km} km<br>"
            f"ETA: {eta_minutes} min",
            max_width=250,
        ),
        tooltip=hospital_name,
        icon=folium.Icon(color="green", icon="plus-sign"),
    ).add_to(m)

    # Route polyline (blue)
    if route_coords and len(route_coords) > 1:
        folium.PolyLine(
            locations=route_coords,
            color="#3B82F6",
            weight=5,
            opacity=0.85,
            tooltip=f"Shortest Route — {distance_km} km | ETA: {eta_minutes} min",
        ).add_to(m)

    return m._repr_html_()


def _get_severity_badge_class(severity: str) -> str:
    """Return a CSS class name based on severity level."""
    return {
        "Fatal Injury":   "badge-fatal",
        "Serious Injury": "badge-serious",
        "Slight Injury":  "badge-slight",
    }.get(severity, "badge-serious")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the accident input form."""
    return render_template("index.html", options=FORM_OPTIONS)


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Receive form submission, run the full pipeline, render results.
    POST body: HTML form fields for accident scenario.
    """
    global _last_map_html

    # ── Collect form inputs ───────────────────────────────────────────────────
    accident_inputs = {
        "Day_of_week":             request.form.get("day_of_week", "Monday"),
        "Weather_conditions":      request.form.get("weather", "Normal"),
        "Light_conditions":        request.form.get("light", "Daylight"),
        "Road_surface_conditions": request.form.get("road_surface", "Dry"),
        "Cause_of_accident":       request.form.get("cause", "Overtaking"),
        "Type_of_collision":       request.form.get("collision_type", "Vehicle with vehicle collision"),
        "Number_of_vehicles_involved": int(request.form.get("vehicles", 2)),
        "Number_of_casualties":        int(request.form.get("casualties", 2)),
    }

    accident_lat     = float(request.form.get("accident_lat", DEFAULT_ACCIDENT_LAT))
    accident_lon     = float(request.form.get("accident_lon", DEFAULT_ACCIDENT_LON))
    junction         = int(request.form.get("junction", 3))
    strategy         = request.form.get("routing_strategy", "capacity")
    selected_hospital= request.form.get("selected_hospital", None) if strategy == "manual" else None

    # ── Step 1: Severity Prediction ───────────────────────────────────────────
    severity = predict(accident_inputs)

    # ── Step 2: Hospital Recommendation ──────────────────────────────────────
    best_hospital = recommend_hospital(
        severity=severity,
        strategy=strategy,
        selected_hospital=selected_hospital,
        start_lat=accident_lat,
        start_lon=accident_lon,
    )
    all_hospitals = get_all_hospitals_ranked(
        severity=severity,
        strategy=strategy,
        selected_hospital=selected_hospital,
        start_lat=accident_lat,
        start_lon=accident_lon,
    )

    hospital_name = best_hospital["Hospital"]
    hospital_lat  = float(best_hospital["Latitude"])
    hospital_lon  = float(best_hospital["Longitude"])
    icu_beds      = int(best_hospital["ICU_Beds"])
    trauma_center = best_hospital["Trauma_Center"]
    hosp_score    = round(float(best_hospital["Score"]), 2)

    # ── Step 3: Traffic Prediction ────────────────────────────────────────────
    now = datetime.datetime.now()
    traffic_features = [[now.hour, now.day, now.month, now.weekday(), junction]]
    traffic = round(predict_traffic(traffic_features), 2)

    # ── Step 4: Dijkstra Route ────────────────────────────────────────────────
    route, route_coords, distance_m, eta = calculate_route(
        traffic   = traffic,
        start_lat = accident_lat,
        start_lon = accident_lon,
        dest_lat  = hospital_lat,
        dest_lon  = hospital_lon,
    )

    distance_km  = round(distance_m / 1000, 2)
    eta_minutes  = round(eta, 2)
    route_nodes  = len(route)

    # ── Step 5: Generate Folium Map ───────────────────────────────────────────
    map_html = _build_folium_map(
        start_lat    = accident_lat,
        start_lon    = accident_lon,
        dest_lat     = hospital_lat,
        dest_lon     = hospital_lon,
        route_coords = route_coords,
        hospital_name= hospital_name,
        distance_km  = distance_km,
        eta_minutes  = eta_minutes,
        severity     = severity,
    )
    _last_map_html = map_html

    # ── Ranked hospitals for sidebar table ────────────────────────────────────
    hospital_table = all_hospitals[
        ["Hospital", "ICU_Beds", "Trauma_Center", "Score", "Latitude", "Longitude", "Distance_km"]
    ].to_dict(orient="records")

    # ── Render results page ───────────────────────────────────────────────────
    return render_template(
        "result.html",
        # Inputs
        accident_inputs  = accident_inputs,
        accident_lat     = accident_lat,
        accident_lon     = accident_lon,
        traffic_junction = junction,
        routing_strategy = strategy,
        # Severity
        severity         = severity,
        severity_class   = _get_severity_badge_class(severity),
        # Hospital
        hospital_name    = hospital_name,
        hospital_lat     = hospital_lat,
        hospital_lon     = hospital_lon,
        icu_beds         = icu_beds,
        trauma_center    = trauma_center,
        hosp_score       = hosp_score,
        hospital_table   = hospital_table,
        # Traffic
        traffic          = traffic,
        # Route
        distance_km      = distance_km,
        eta_minutes      = eta_minutes,
        route_nodes      = route_nodes,
        route_coords_json= json.dumps(route_coords),
        # Map
        map_html         = map_html,
        # Timestamp
        timestamp        = now.strftime("%Y-%m-%d %H:%M:%S"),
    )


@app.route("/api/route_to_hospital", methods=["POST"])
def api_route_to_hospital():
    """
    AJAX endpoint: Dynamic route recalculation when user selects a specific hospital from the table.
    """
    data = request.get_json() or {}
    accident_lat  = float(data.get("accident_lat", DEFAULT_ACCIDENT_LAT))
    accident_lon  = float(data.get("accident_lon", DEFAULT_ACCIDENT_LON))
    dest_lat      = float(data.get("dest_lat", 16.5062))
    dest_lon      = float(data.get("dest_lon", 80.6480))
    hospital_name = data.get("hospital_name", "Selected Hospital")
    traffic       = float(data.get("traffic", 15.0))

    route, route_coords, distance_m, eta = calculate_route(
        traffic   = traffic,
        start_lat = accident_lat,
        start_lon = accident_lon,
        dest_lat  = dest_lat,
        dest_lon  = dest_lon,
    )

    distance_km = round(distance_m / 1000, 2)
    eta_minutes = round(eta, 2)

    return jsonify({
        "success": True,
        "hospital_name": hospital_name,
        "distance_km": distance_km,
        "eta_minutes": eta_minutes,
        "route_nodes": len(route),
        "route_coords": route_coords,
    })


@app.route("/map")
def full_map():
    """Serve the last generated route map in full screen."""
    if _last_map_html is None:
        return "<h3>No map generated yet. Please submit the form first.</h3>"
    return _last_map_html


@app.errorhandler(500)
def server_error(e):
    return render_template("index.html", options=FORM_OPTIONS,
                           error=f"Pipeline error: {str(e)}"), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  🚑  Smart Ambulance System — Web Dashboard")
    print("  URL: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)