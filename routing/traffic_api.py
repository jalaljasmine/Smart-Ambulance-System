import urllib.request
import urllib.parse
import json
import os

def get_traffic_route(start_coords, end_coords, api_key=None):
    """
    Calls the TomTom Routing API to calculate a traffic-aware route.
    Args:
        start_coords (tuple): (latitude, longitude) for start point
        end_coords (tuple): (latitude, longitude) for end point
        api_key (str): TomTom API Key. If None, it will look up TOMTOM_API_KEY environment variable.
    Returns:
        dict: containing:
            'success': True/False
            'distance_m': distance in meters
            'time_min': estimated time in minutes (including traffic)
            'traffic_delay_min': traffic delay in minutes
            'route_points': list of (lat, lon) coordinates representing the path
            'error': error message if success is False
    """
    if not api_key:
        api_key = os.environ.get("TOMTOM_API_KEY")
        
    if not api_key:
        return {
            "success": False,
            "error": "TomTom API Key is missing. Add TOMTOM_API_KEY to your environment or .env file."
        }
        
    # Format locations parameter: lat,lon:lat,lon
    locations = f"{start_coords[0]},{start_coords[1]}:{end_coords[0]},{end_coords[1]}"
    
    # Base URL
    base_url = f"https://api.tomtom.com/routing/1/calculateRoute/{locations}/json"
    
    # Query parameters
    params = {
        "key": api_key,
        "traffic": "true",
        "departAt": "now"
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'SmartAmbulanceSystem/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            
        if "routes" not in data or not data["routes"]:
            return {
                "success": False,
                "error": "No route found in API response."
            }
            
        route = data["routes"][0]
        summary = route["summary"]
        
        distance_m = summary.get("lengthInMeters", 0)
        time_sec = summary.get("travelTimeInSeconds", 0)
        delay_sec = summary.get("trafficDelayInSeconds", 0)
        
        # Extract path geometry (coordinates of the route)
        route_points = []
        if "legs" in route and route["legs"]:
            for leg in route["legs"]:
                if "points" in leg:
                    for pt in leg["points"]:
                        route_points.append((pt["latitude"], pt["longitude"]))
                        
        return {
            "success": True,
            "distance_m": distance_m,
            "time_min": time_sec / 60.0,
            "traffic_delay_min": delay_sec / 60.0,
            "route_points": route_points
        }
        
    except urllib.error.HTTPError as e:
        try:
            err_data = json.loads(e.read().decode())
            err_msg = err_data.get("detailedError", {}).get("message", str(e))
        except Exception:
            err_msg = str(e)
        return {
            "success": False,
            "error": f"TomTom API HTTP Error: {err_msg}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"TomTom API Connection Error: {str(e)}"
        }
