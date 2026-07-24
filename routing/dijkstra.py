import os
import osmnx as ox
import networkx as nx

from routing.check_traffic import predict_traffic
from routing.traffic_api import get_traffic_route


def calculate_route(start_coords, end_coords, traffic=None):
    """
    Calculates the best route between start_coords and end_coords.
    Attempts to use the TomTom Traffic API if TOMTOM_API_KEY is in the environment.
    Falls back to offline Dijkstra routing using a pre-saved OpenStreetMap graph.
    
    Args:
        start_coords (tuple): (lat, lon)
        end_coords (tuple): (lat, lon)
        traffic (float): Option prediction output for offline mode.
        
    Returns:
        tuple: (route, distance_m, travel_time_min)
            - route: list of (lat, lon) coordinates or node IDs
            - distance_m: distance in meters
            - travel_time_min: estimated travel time in minutes
    """
    # 1. Try real-time TomTom Traffic API first
    api_key = os.environ.get("TOMTOM_API_KEY")
    if api_key:
        print("Attempting to fetch real-time traffic-aware route via TomTom API...")
        result = get_traffic_route(start_coords, end_coords, api_key)
        if result["success"]:
            print("Successfully retrieved live traffic route!")
            return result["route_points"], result["distance_m"], result["time_min"]
        else:
            print(f"TomTom API failed: {result['error']}. Falling back to offline Dijkstra.")
    else:
        print("No TOMTOM_API_KEY environment variable found. Using offline Dijkstra routing.")

    # 2. Offline Dijkstra Fallback
    print("Loading saved graph...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    graph_path = os.path.join(project_dir, "datasets", "vijayawada.graphml")
    
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Offline graph file not found at {graph_path}. Please run download_roads.py first.")
        
    G = ox.load_graphml(graph_path)
    print("Graph loaded successfully!")

    # Find nearest graph nodes
    # Note: OSMnx expects (lon, lat) or x, y for distance calculation
    orig_node = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
    dest_node = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])

    # Dijkstra shortest path based on length
    route = nx.shortest_path(G, orig_node, dest_node, weight="length")
    distance = nx.shortest_path_length(G, orig_node, dest_node, weight="length")

    # Traffic Prediction Fallback
    if traffic is None:
        import datetime
        now = datetime.datetime.now()
        # [Hour, Day, Month, Weekday, Junction]
        input_data = [[now.hour, now.day, now.month, now.weekday(), 3]]
        traffic = predict_traffic(input_data)

    if traffic < 10:
        speed = 40  # km/h
    elif traffic < 20:
        speed = 30  # km/h
    else:
        speed = 20  # km/h

    time_minutes = (distance / 1000) / speed * 60

    return route, distance, time_minutes


if __name__ == "__main__":
    start = (16.5062, 80.6480)
    end = (16.5150, 80.6300)
    route, distance, eta = calculate_route(start, end)
    print(f"\nRoute Nodes: {len(route)}")
    print(f"Distance: {round(distance / 1000, 2)} km")
    print(f"Estimated Time: {round(eta, 2)} minutes")