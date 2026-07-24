import os
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

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

# Set page configuration
st.set_page_config(
    page_title="Smart Ambulance System",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #0f1115;
        color: #e9ecef;
    }
    div[data-testid="stSidebar"] {
        background-color: #161920;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 15px;
    }
    .metric-title {
        color: #adb5bd;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #3b5bdb;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 5px;
    }
    .hospital-title {
        color: #e03131;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .lock-badge {
        display: inline-block;
        background: #2f9e44;
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        margin-top: 8px;
        letter-spacing: 0.5px;
    }
    .lock-badge.warn {
        background: #e67700;
    }
    </style>
""", unsafe_allow_html=True)

# Cache graph resource for converting node IDs to coordinates
@st.cache_resource
def load_graph():
    import osmnx as ox
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    graph_path = os.path.join(project_dir, "datasets", "vijayawada.graphml")
    if os.path.exists(graph_path):
        return ox.load_graphml(graph_path)
    return None

# Cache LockManager as a singleton — persists across Streamlit reruns
@st.cache_resource
def load_lock_manager():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    csv_path = os.path.join(project_dir, "datasets", "hospitals.csv")
    return LockManager(csv_path)

G = load_graph()
lock_manager = load_lock_manager()

def get_route_coords(route_list):
    if not route_list:
        return []
    if isinstance(route_list[0], (list, tuple)):
        return route_list
    # Convert node IDs to coordinates
    coords = []
    if G is not None:
        for node in route_list:
            node_data = G.nodes[node]
            coords.append((node_data['y'], node_data['x']))
    return coords

# ----------------- SIDEBAR: INPUT CONTROLS -----------------
st.sidebar.title("🚨 Emergency Controls")

# Input Mode Selector
severity_mode = st.sidebar.radio(
    "Accident Severity Assessment Mode",
    ["Manual Input", "Machine Learning Model Prediction"]
)

if severity_mode == "Manual Input":
    manual_severity = st.sidebar.selectbox(
        "Select Severity Level",
        ["Slight Injury", "Serious Injury", "Fatal Injury"],
        index=1
    )
    severity = manual_severity
else:
    st.sidebar.markdown("### Accident Feature Details")
    day = st.sidebar.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)
    age = st.sidebar.selectbox("Driver Age Band", ["18-30", "31-50", "Over 51", "Under 18", "Unknown"], index=0)
    light = st.sidebar.selectbox("Light Conditions", ["Daylight", "Darkness - lights lit", "Darkness - lights unlit", "Darkness - no lighting"], index=0)
    weather = st.sidebar.selectbox("Weather Conditions", ["Normal", "Raining", "Raining and Windy", "Windy", "Snowing", "Fog or mist", "Other"], index=0)
    vehicles = st.sidebar.number_input("Number of Vehicles Involved", min_value=1, max_value=10, value=2)
    casualties = st.sidebar.number_input("Number of Casualties", min_value=1, max_value=20, value=1)
    
    input_features = {
        "Day_of_week": day,
        "Age_band_of_driver": age,
        "Light_conditions": light,
        "Weather_conditions": weather,
        "Number_of_vehicles_involved": vehicles,
        "Number_of_casualties": casualties
    }
    
    # Predict automatically for real-time interactivity
    severity = predict(input_features)
    st.sidebar.success(f"🎯 Predicted Severity: **{severity}**")

st.sidebar.markdown("---")
# TomTom API Key display / input override
st.sidebar.markdown("### Routing Configuration")
tomtom_key = os.environ.get("TOMTOM_API_KEY")
if tomtom_key:
    st.sidebar.success("🔑 TomTom Traffic API: Connected")
else:
    st.sidebar.warning("⚠️ Using Offline Dijkstra (No API Key found)")
    api_key_input = st.sidebar.text_input("Enter TomTom API Key (Optional)", type="password")
    if api_key_input:
        os.environ["TOMTOM_API_KEY"] = api_key_input
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Traffic Forecast Settings")
import datetime
current_time = datetime.datetime.now()

traffic_junction = st.sidebar.selectbox("Simulated Junction ID", [1, 2, 3, 4], index=2)
traffic_hour = st.sidebar.slider("Hour of Day", 0, 23, value=current_time.hour)
traffic_day = st.sidebar.slider("Day of Month", 1, 31, value=current_time.day)
traffic_month = st.sidebar.slider("Month of Year", 1, 12, value=current_time.month)

# Calculate Weekday automatically
try:
    sim_date = datetime.date(current_time.year, traffic_month, traffic_day)
    traffic_weekday = sim_date.weekday()
except Exception:
    traffic_weekday = current_time.weekday()

weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
st.sidebar.caption(f"Simulated Day: **{weekday_names[traffic_weekday]}**")

# ----------------- MAIN LAYOUT -----------------
st.title("🚑 Smart Ambulance Routing & Traffic Optimization")
st.markdown("Click anywhere on the map to set the accident location in real-time.")

# Initialize session state for accident coordinates and active lock
if "accident_coords" not in st.session_state:
    st.session_state.accident_coords = (16.5062, 80.6480)
if "active_lock_id" not in st.session_state:
    st.session_state.active_lock_id = None
if "ambulance_id" not in st.session_state:
    import uuid
    st.session_state.ambulance_id = f"amb_{uuid.uuid4().hex[:6]}"

# Expire any stale locks before making new recommendations
import datetime
lock_manager.expire_stale_locks(current_time.timestamp() / 60.0)

# 1. Recommended Hospital with reservation lock (falls back through leaderboard if full)
rec_hospital, active_lock = recommend_hospital_with_lock(
    severity=severity,
    lock_manager=lock_manager,
    ambulance_id=st.session_state.ambulance_id,
    current_time=current_time.timestamp() / 60.0,
    start_coords=st.session_state.accident_coords,
)

# Track the current active lock in session state
if active_lock:
    # Release any previous lock held by this session before assigning the new one
    if st.session_state.active_lock_id and st.session_state.active_lock_id != active_lock.lock_id:
        lock_manager.release_lock(st.session_state.active_lock_id)
    st.session_state.active_lock_id = active_lock.lock_id

dest_coords = (rec_hospital["Latitude"], rec_hospital["Longitude"])

# 2. Predicted Traffic Volume
traffic_input = [[traffic_hour, traffic_day, traffic_month, traffic_weekday, traffic_junction]]
traffic_prediction = predict_traffic(traffic_input)

# 3. Calculate Route
route, distance_m, time_min = calculate_route(
    st.session_state.accident_coords, 
    dest_coords, 
    traffic=traffic_prediction
)

# Convert route to coordinates for drawing
route_coords = get_route_coords(route)

# Create layout columns
col_map, col_info = st.columns([2, 1])

with col_map:
    st.markdown("### 🗺️ Emergency Navigation Map")
    
    # Initialize Folium Map centered on Vijayawada
    m = folium.Map(location=[16.5062, 80.6480], zoom_start=13, tiles="cartodbpositron")
    m.add_child(folium.LatLngPopup()) # Click popup helper
    
    # Load all hospitals
    hospitals_df = pd.read_csv("datasets/hospitals.csv")
    
    # Render all hospitals as pins — show live available beds from LockManager
    for idx, row in hospitals_df.iterrows():
        h_name = row["Hospital"]
        pool = lock_manager.pools.get(h_name)
        avail = pool.available_beds if pool else row["ICU_Beds"]
        total = pool.total_beds if pool else row["ICU_Beds"]
        beds_label = f"Beds Available: {avail}/{total}"
        
        if h_name == rec_hospital["Hospital"]:
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=f"<b>[RESERVED] {h_name}</b><br>{beds_label}<br>Staff: {row['Emergency_Staff']}<br>🔒 Bed pre-reserved",
                icon=folium.Icon(color="red", icon="plus-sign")
            ).add_to(m)
        else:
            color = "blue" if avail > 0 else "gray"
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=f"<b>{h_name}</b><br>{beds_label}<br>Staff: {row['Emergency_Staff']}",
                icon=folium.Icon(color=color, icon="plus-sign")
            ).add_to(m)
            
    # Render Accident marker in orange
    folium.Marker(
        location=st.session_state.accident_coords,
        popup=f"<b>Accident Site</b><br>Lat: {round(st.session_state.accident_coords[0], 4)}<br>Lon: {round(st.session_state.accident_coords[1], 4)}",
        icon=folium.Icon(color="orange", icon="exclamation-sign")
    ).add_to(m)
    
    # Render Route line in red if route exists and start != end
    if route_coords and st.session_state.accident_coords != dest_coords:
        folium.PolyLine(
            route_coords,
            color="red",
            weight=5,
            opacity=0.8,
            tooltip="Ambulance Routing Path"
        ).add_to(m)
        
    # Render Map and get clicks
    map_data = st_folium(m, width="100%", height=550)
    
    # Check if map clicked
    if map_data and map_data.get("last_clicked"):
        clicked = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])
        if clicked != st.session_state.accident_coords:
            st.session_state.accident_coords = clicked
            st.rerun()

with col_info:
    st.markdown("### 📋 Dispatch & Route Metrics")
    
    # Get live bed pool info
    pool = lock_manager.pools.get(rec_hospital["Hospital"])
    avail_beds = pool.available_beds if pool else rec_hospital["ICU_Beds"]
    total_beds = pool.total_beds if pool else rec_hospital["ICU_Beds"]
    locked_beds = pool.locked_beds if pool else 0
    occupied_beds = pool.occupied_beds if pool else 0
    
    lock_status_html = ""
    if active_lock:
        lock_status_html = '<span class="lock-badge">🔒 BED PRE-RESERVED</span>'
    else:
        lock_status_html = '<span class="lock-badge warn">⚠️ NO BEDS — FALLBACK HOSPITAL</span>'
    
    # Hospital Card with live lock status
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🚑 Dispatched Hospital</div>
            <div class="hospital-title">{rec_hospital['Hospital']}</div>
            {lock_status_html}
            <p style="margin-top: 10px; margin-bottom: 0;">
                <b>Trauma Center:</b> {rec_hospital['Trauma_Center']}<br>
                <b>Total ICU Beds:</b> {total_beds}<br>
                <b>🟢 Available Beds:</b> {avail_beds}<br>
                <b>🔒 Reserved (In Transit):</b> {locked_beds}<br>
                <b>🔴 Occupied:</b> {occupied_beds}<br>
                <b>Emergency Staff:</b> {rec_hospital['Emergency_Staff']}<br>
                <b>Target Coordinates:</b> {round(dest_coords[0], 4)}, {round(dest_coords[1], 4)}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Lock Actions
    if active_lock and st.session_state.active_lock_id:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Mark Arrived", help="Convert the reservation to an occupied bed on arrival", use_container_width=True):
                lock_manager.convert_to_occupied(st.session_state.active_lock_id)
                st.session_state.active_lock_id = None
                st.session_state.ambulance_id = f"amb_{__import__('uuid').uuid4().hex[:6]}"
                st.success("Bed marked as occupied. Ambulance dispatched!")
                st.rerun()
        with col_b:
            if st.button("❌ Release Lock", help="Cancel the reservation and free the bed", use_container_width=True):
                lock_manager.release_lock(st.session_state.active_lock_id)
                st.session_state.active_lock_id = None
                st.session_state.ambulance_id = f"amb_{__import__('uuid').uuid4().hex[:6]}"
                st.warning("Reservation released. Bed is now free.")
                st.rerun()
    
    # Distance Card
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📏 Total Routing Distance</div>
            <div class="metric-value">{round(distance_m / 1000, 2)} km</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Travel Time Card
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⏱️ Estimated Travel Time</div>
            <div class="metric-value">{round(time_min, 1)} minutes</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Severity & Traffic Summary
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📊 Pipeline Parameters</div>
            <p style="margin-top: 10px; margin-bottom: 0;">
                <b>Accident Severity:</b> {severity}<br>
                <b>Local Traffic Forecast:</b> {round(traffic_prediction, 1)} vehicles/hr<br>
                <b>Coordinates Selected:</b> {round(st.session_state.accident_coords[0], 4)}, {round(st.session_state.accident_coords[1], 4)}
            </p>
        </div>
    """, unsafe_allow_html=True)
