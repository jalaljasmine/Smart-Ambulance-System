"""
routing/dijkstra.py — Smart Ambulance System
=============================================
Loads the pre-built Vijayawada road network (GraphML format) and
computes the shortest ambulance route between an accident location
and the recommended hospital using Dijkstra's algorithm via NetworkX.

Key features:
- Loads vijayawada.graphml once (no repeated downloads)
- Accepts dynamic start and destination coordinates
- Uses pre-predicted traffic volume to estimate travel speed
- Returns route node list, distance (metres), and ETA (minutes)

Bug fixes applied (vs. original):
  1. Removed duplicate predict_traffic() call inside calculate_route()
  2. Added dest_lat/dest_lon parameters — hospital coordinates from
     hospital_score.py are now used instead of hardcoded values
  3. Fixed standalone __main__ call signature
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# pyrefly: ignore [missing-import]
import osmnx as ox
import networkx as nx

# Allow import from project root regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    GRAPHML_PATH,
    DEFAULT_ACCIDENT_LAT,
    DEFAULT_ACCIDENT_LON,
    get_speed_from_traffic,
)

# ── Module-level graph cache (loaded once per session) ────────────────────────
_graph = None


def _load_graph():
    """Load the Vijayawada road network graph from disk (cached after first load)."""
    global _graph
    if _graph is None:
        print("  [Router] Loading road network from disk...")
        _graph = ox.load_graphml(GRAPHML_PATH)
        print("  [Router] Road network loaded successfully.")
    return _graph


def calculate_route(
    traffic: float,
    start_lat: float = DEFAULT_ACCIDENT_LAT,
    start_lon: float = DEFAULT_ACCIDENT_LON,
    dest_lat:  float = None,
    dest_lon:  float = None,
) -> tuple:
    """
    Calculate the shortest route from the accident location to the hospital.

    Args:
        traffic   (float): Predicted vehicle count (from check_traffic.py).
                           Used to estimate ambulance speed.
        start_lat (float): Accident location latitude.  Default: Vijayawada demo.
        start_lon (float): Accident location longitude. Default: Vijayawada demo.
        dest_lat  (float): Hospital latitude.  If None, uses default hospital coords.
        dest_lon  (float): Hospital longitude. If None, uses default hospital coords.

    Returns:
        tuple: (route, distance_metres, eta_minutes)
            - route           : list of OSMnx node IDs along the shortest path
            - distance_metres : total path length in metres (float)
            - eta_minutes     : estimated travel time in minutes (float)
    """
    G = _load_graph()

    # Default destination: Government General Hospital, Vijayawada
    if dest_lat is None or dest_lon is None:
        dest_lat = 16.5150
        dest_lon = 80.6300

    # Find nearest graph nodes to start and destination coordinates
    orig_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    dest_node = ox.distance.nearest_nodes(G, dest_lon, dest_lat)

    # Dijkstra shortest path by road length (NetworkX uses Dijkstra internally)
    route = nx.shortest_path(G, orig_node, dest_node, weight="length")

    distance_m = nx.shortest_path_length(
        G, orig_node, dest_node, weight="length"
    )

    # Estimate ambulance speed based on predicted traffic
    speed_kmh = get_speed_from_traffic(traffic)

    # ETA = distance (km) / speed (km/h) × 60 → minutes
    distance_km = distance_m / 1000
    eta_minutes = (distance_km / speed_kmh) * 60

    # Extract route coordinates for map visualization
    route_coords = [
        (G.nodes[node]["y"], G.nodes[node]["x"])
        for node in route
    ]

    return route, route_coords, distance_m, eta_minutes


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from routing.check_traffic import predict_traffic
    from config import DEFAULT_TRAFFIC_INPUT

    # Predict traffic first, then pass to route calculator
    traffic_val = predict_traffic(DEFAULT_TRAFFIC_INPUT)
    print(f"Predicted Traffic: {traffic_val:.2f} vehicles/hour")

    route, route_coords, dist_m, eta = calculate_route(
        traffic=traffic_val,
        start_lat=DEFAULT_ACCIDENT_LAT,
        start_lon=DEFAULT_ACCIDENT_LON,
    )

    print(f"Route nodes   : {len(route)}")
    print(f"Distance      : {dist_m / 1000:.2f} km")
    print(f"Estimated ETA : {eta:.2f} minutes")