import os
import math
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional

# ----------------- MONKEYPATCHING OSMNX FOR CACHING -----------------
# We preload the graph and monkeypatch ox.load_graphml to avoid reading the 11MB file on every route calculation.
import osmnx as ox
import networkx as nx

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
graph_path = os.path.join(project_dir, "datasets", "vijayawada.graphml")

print("Pre-loading road network graph for simulation caching...")
if not os.path.exists(graph_path):
    raise FileNotFoundError(f"Vijayawada graphml file not found at {graph_path}")
_cached_graph = ox.load_graphml(graph_path)
print("Graph loaded and cached successfully!")

_orig_load_graphml = ox.load_graphml

def cached_load_graphml(filepath, *args, **kwargs):
    return _cached_graph

ox.load_graphml = cached_load_graphml
# Precompute valid (routable) nodes: only those in the largest strongly connected component.
# This prevents NetworkXNoPath errors when randomly sampled nodes have no route to hospitals.
_largest_scc_nodes = set(max(nx.strongly_connected_components(_cached_graph), key=len))
_routable_nodes = [(n, d) for n, d in _cached_graph.nodes(data=True) if n in _largest_scc_nodes]
print(f"Routable node pool: {len(_routable_nodes)} / {len(_cached_graph.nodes)} nodes (largest SCC).")
# ---------------------------------------------------------------------

from models.predict_severity import predict
from models.hospital_score import get_sorted_hospitals, recommend_hospital_with_lock, recommend_hospital
from routing.dijkstra import calculate_route

def get_route_coords(route_nodes: List[Any]) -> List[Tuple[float, float]]:
    """Converts a list of graph node IDs to a list of (latitude, longitude) coordinate tuples."""
    coords = []
    for node in route_nodes:
        node_data = _cached_graph.nodes[node]
        coords.append((node_data['y'], node_data['x']))
    return coords

def interpolate_position(route_coords: List[Tuple[float, float]], fraction: float) -> Tuple[float, float]:
    """Interpolates coordinates along a route based on elapsed travel time fraction."""
    if not route_coords:
        return (16.5062, 80.6480)
    if fraction <= 0.0:
        return route_coords[0]
    if fraction >= 1.0:
        return route_coords[-1]
    
    n = len(route_coords)
    idx = fraction * (n - 1)
    low_idx = int(math.floor(idx))
    high_idx = int(math.ceil(idx))
    
    if low_idx == high_idx:
        return route_coords[low_idx]
    
    t = idx - low_idx
    p1 = route_coords[low_idx]
    p2 = route_coords[high_idx]
    
    lat = p1[0] + t * (p2[0] - p1[0])
    lon = p1[1] + t * (p2[1] - p1[1])
    return (lat, lon)

def generate_calls(num_calls: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generates a reproducible list of ambulance calls.
    Returns:
        List[Dict]: keys: call_id, arrival_time, start_coords, features, occupancy_duration
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Sample only from the largest strongly connected component (routable nodes)
    calls = []
    
    for i in range(num_calls):
        call_id = i + 1
        # Randomized arrival time between 0 and 40 minutes (keeps it dense for lock contention)
        arrival_time = random.uniform(0.0, 40.0)
        
        # Pick a random node guaranteed to be in the largest SCC (routable to all hospitals)
        _, node_data = random.choice(_routable_nodes)
        start_coords = (node_data['y'], node_data['x'])
        
        # Generate random features for ML model severity prediction
        features = {
            "Day_of_week": random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
            "Age_band_of_driver": random.choice(["18-30", "31-50", "Over 51", "Under 18"]),
            "Light_conditions": random.choice(["Daylight", "Darkness - lights lit", "Darkness - lights unlit"]),
            "Weather_conditions": random.choice(["Normal", "Raining", "Raining and Windy", "Windy"]),
            "Number_of_vehicles_involved": random.randint(1, 4),
            "Number_of_casualties": random.randint(1, 4)
        }
        
        # Randomized bed occupancy (treatment) duration between 40 and 100 minutes
        occupancy_duration = random.uniform(40.0, 100.0)
        
        calls.append({
            "call_id": call_id,
            "arrival_time": arrival_time,
            "start_coords": start_coords,
            "features": features,
            "occupancy_duration": occupancy_duration
        })
        
    # Sort calls by arrival time
    calls = sorted(calls, key=lambda x: x["arrival_time"])
    return calls


class AmbulanceAgent:
    """
    Tracks the state and telemetry of an ambulance during simulation.
    """
    def __init__(self, agent_id: str, call_data: Dict[str, Any], severity: str):
        self.agent_id: str = agent_id
        self.call_id: int = call_data["call_id"]
        self.call_arrival_time: float = call_data["arrival_time"]
        self.start_coords: Tuple[float, float] = call_data["start_coords"]
        self.current_coords: Tuple[float, float] = call_data["start_coords"]
        self.severity: str = severity
        self.occupancy_duration: float = call_data["occupancy_duration"]
        
        # Routing and destination state
        self.status: str = "IDLE"  # IDLE, TRAVELLING, WAITING, ADMITTED, COMPLETED
        self.dest_hospital: str = ""
        self.dest_coords: Tuple[float, float] = (0.0, 0.0)
        self.route_nodes: List[Any] = []
        self.route_coords: List[Tuple[float, float]] = []
        self.total_distance: float = 0.0
        self.total_travel_time: float = 0.0
        self.elapsed_travel_time: float = 0.0
        
        # Reservation and timing tracking
        self.lock_id: Optional[str] = None
        self.arrival_time: Optional[float] = None
        self.admission_time: Optional[float] = None
        self.release_time: Optional[float] = None
        self.waiting_time: float = 0.0
        self.time_to_admission: float = 0.0
        self.lock_swaps: int = 0
        self.re_eval_timer: float = 0.0

    def update_route(self, dest_hospital: str, dest_coords: Tuple[float, float], 
                     route_nodes: List[Any], distance: float, travel_time: float):
        """Updates route information when dispatching or swapping."""
        self.dest_hospital = dest_hospital
        self.dest_coords = dest_coords
        self.route_nodes = route_nodes
        self.route_coords = get_route_coords(route_nodes)
        self.total_distance = distance
        self.total_travel_time = travel_time
        self.elapsed_travel_time = 0.0


def run_simulation(calls: List[Dict[str, Any]], 
                   mode: str = "LOCKED", 
                   bed_scale_factor: float = 0.05, 
                   re_eval_interval: float = 2.0, 
                   max_time: float = 400.0, 
                   dt: float = 0.1,
                   scoring_mode: str = "original") -> Dict[str, Any]:
    """
    Runs a discrete-event-style time-stepped simulation of multi-ambulance dispatching.
    
    Args:
        calls (List[Dict]): Pre-generated call sequence.
        mode (str): "LOCKED" (with coordination layer & swaps) or "BASELINE" (no lock).
        bed_scale_factor (float): Scaler for hospital bed capacities (simulates constraints).
        re_eval_interval (float): Time interval (mins) at which travelling ambulances re-evaluate.
        max_time (float): Safety cap for simulation time (mins).
        dt (float): Timestep size (mins).
        scoring_mode (str): ``"original"`` or ``"normalized"`` — passed to hospital scoring.
        
    Returns:
        Dict: Compiled metrics of the simulation run.
    """
    csv_path = os.path.join(project_dir, "datasets", "hospitals.csv")
    
    # Import LockManager here to avoid circularity
    from coordination.lock_manager import LockManager
    lock_manager = LockManager(csv_path)
    
    # Scale hospital bed counts
    for h_name, pool in lock_manager.pools.items():
        pool.total_beds = max(1, int(round(pool.total_beds * bed_scale_factor)))
        pool.locked_beds = 0
        pool.occupied_beds = 0
        
    print(f"\n--- Starting {mode} Simulation (Beds Scaled: {bed_scale_factor}) ---")
    for h_name, pool in lock_manager.pools.items():
        print(f"  {h_name}: {pool.total_beds} beds")

    active_agents: List[AmbulanceAgent] = []
    completed_agents: List[AmbulanceAgent] = []
    hospital_queues: Dict[str, List[AmbulanceAgent]] = {h_name: [] for h_name in lock_manager.pools}
    
    # Tracks pending bed releases: List of Tuple[release_time, hospital_name, agent_id, lock_id_or_none]
    bed_releases: List[Tuple[float, str, str, Optional[str]]] = []
    
    # Tracks how many times an ambulance arrives at a hospital with 0 available beds
    overcommit_events = 0
    
    current_time = 0.0
    spawned_call_count = 0
    
    while current_time <= max_time:
        # Check if all calls are generated and all agents are COMPLETED
        if spawned_call_count >= len(calls) and not active_agents:
            break
            
        # 1. Process Bed Releases
        active_releases = [r for r in bed_releases if r[0] <= current_time]
        for release in active_releases:
            rel_time, h_name, agent_id, lock_id = release
            pool = lock_manager.pools[h_name]
            
            # Decrement bed count
            if mode == "LOCKED":
                if lock_id:
                    lock_manager.release_lock(lock_id)
                else:
                    # Queued fallback case without lock
                    pool.occupied_beds = max(0, pool.occupied_beds - 1)
            else:
                pool.occupied_beds = max(0, pool.occupied_beds - 1)
                
            # If a queue is waiting at this hospital, admit the next ambulance
            queue = hospital_queues[h_name]
            if queue:
                next_agent = queue.pop(0)
                next_agent.status = "ADMITTED"
                next_agent.admission_time = current_time
                next_agent.waiting_time = current_time - next_agent.arrival_time
                next_agent.time_to_admission = current_time - next_agent.call_arrival_time
                next_agent.release_time = current_time + next_agent.occupancy_duration
                
                # Increment occupancy
                if mode == "LOCKED":
                    # If this agent had a lock that expired, attempt to claim a new one, else direct occupy
                    if next_agent.lock_id:
                        lock_manager.convert_to_occupied(next_agent.lock_id)
                    else:
                        pool.occupied_beds += 1
                else:
                    pool.occupied_beds += 1
                    
                # Schedule release
                bed_releases.append((next_agent.release_time, h_name, next_agent.agent_id, next_agent.lock_id))
                
        # Remove processed releases
        bed_releases = [r for r in bed_releases if r[0] > current_time]
        
        # 2. Expire Stale Locks (LOCKED mode only)
        if mode == "LOCKED":
            lock_manager.expire_stale_locks(current_time)
            
        # 3. Spawn New Calls
        while spawned_call_count < len(calls) and calls[spawned_call_count]["arrival_time"] <= current_time:
            call_data = calls[spawned_call_count]
            spawned_call_count += 1
            
            # Determine Severity
            severity = predict(call_data["features"])
            agent_id = f"amb_{call_data['call_id']}"
            agent = AmbulanceAgent(agent_id, call_data, severity)
            
            # Hospital selection flow
            if mode == "LOCKED":
                # Find hospital and claim reservation lock
                h_row, lock = recommend_hospital_with_lock(
                    severity=severity,
                    lock_manager=lock_manager,
                    ambulance_id=agent_id,
                    current_time=current_time,
                    start_coords=agent.start_coords,
                    buffer=8.0,
                    scoring_mode=scoring_mode
                )
                dest_hospital = h_row["Hospital"]
                dest_coords = (h_row["Latitude"], h_row["Longitude"])
                agent.lock_id = lock.lock_id if lock else None
            else:
                # Baseline: No locks, just recommend hospital
                h_row = recommend_hospital(severity, start_coords=agent.start_coords, scoring_mode=scoring_mode)
                dest_hospital = h_row["Hospital"]
                dest_coords = (h_row["Latitude"], h_row["Longitude"])
                agent.lock_id = None
                
            # Initial routing
            route_nodes, distance, travel_time = calculate_route(agent.start_coords, dest_coords)
            agent.update_route(dest_hospital, dest_coords, route_nodes, distance, travel_time)
            agent.status = "TRAVELLING"
            
            active_agents.append(agent)
            
        # 4. Update Travelling Agents
        still_active = []
        for agent in active_agents:
            if agent.status == "TRAVELLING":
                agent.elapsed_travel_time += dt
                
                # Arrival check
                if agent.elapsed_travel_time >= agent.total_travel_time:
                    agent.arrival_time = current_time
                    pool = lock_manager.pools[agent.dest_hospital]
                    
                    # Admission flow
                    if mode == "LOCKED":
                        lock = lock_manager.locks.get(agent.lock_id) if agent.lock_id else None
                        if lock and lock.status == "HELD":
                            # Successful reserved admission
                            lock_manager.convert_to_occupied(agent.lock_id)
                            agent.status = "ADMITTED"
                            agent.admission_time = current_time
                            agent.waiting_time = 0.0
                            agent.time_to_admission = current_time - agent.call_arrival_time
                            agent.release_time = current_time + agent.occupancy_duration
                            bed_releases.append((agent.release_time, agent.dest_hospital, agent.agent_id, agent.lock_id))
                        else:
                            # Lock expired or no lock claimed at startup (fallback)
                            # Check if a bed is available anyway
                            if pool.available_beds > 0:
                                if lock:
                                    lock.status = "CONVERTED"
                                pool.occupied_beds += 1
                                agent.status = "ADMITTED"
                                agent.admission_time = current_time
                                agent.waiting_time = 0.0
                                agent.time_to_admission = current_time - agent.call_arrival_time
                                agent.release_time = current_time + agent.occupancy_duration
                                bed_releases.append((agent.release_time, agent.dest_hospital, agent.agent_id, agent.lock_id))
                            else:
                                # Overcommitment event! Lock expired or missing, and hospital is full.
                                overcommit_events += 1
                                agent.status = "WAITING"
                                hospital_queues[agent.dest_hospital].append(agent)
                    else:
                        # Baseline mode
                        if pool.available_beds > 0:
                            pool.occupied_beds += 1
                            agent.status = "ADMITTED"
                            agent.admission_time = current_time
                            agent.waiting_time = 0.0
                            agent.time_to_admission = current_time - agent.call_arrival_time
                            agent.release_time = current_time + agent.occupancy_duration
                            bed_releases.append((agent.release_time, agent.dest_hospital, agent.agent_id, None))
                        else:
                            overcommit_events += 1
                            agent.status = "WAITING"
                            hospital_queues[agent.dest_hospital].append(agent)
                    
                    still_active.append(agent)
                    
                else:
                    # Still driving - interpolate coordinates
                    fraction = agent.elapsed_travel_time / agent.total_travel_time
                    agent.current_coords = interpolate_position(agent.route_coords, fraction)
                    
                    # Periodic re-evaluation tick (LOCKED mode only)
                    if mode == "LOCKED":
                        agent.re_eval_timer += dt
                        if agent.re_eval_timer >= re_eval_interval:
                            agent.re_eval_timer = 0.0
                            
                            # Check if a better hospital is available and claimable
                            sorted_hospitals = get_sorted_hospitals(agent.severity, agent.current_coords, scoring_mode=scoring_mode)
                            current_score = sorted_hospitals[sorted_hospitals["Hospital"] == agent.dest_hospital]["Score"].values[0]
                            
                            for _, row in sorted_hospitals.iterrows():
                                h_name = row["Hospital"]
                                h_score = row["Score"]
                                
                                # Better hospital means a higher score and not our current destination
                                if h_name != agent.dest_hospital and h_score > current_score:
                                    pool = lock_manager.pools[h_name]
                                    if pool.available_beds > 0:
                                        # Calculate route to this potential hospital from current coords
                                        h_coords = (row["Latitude"], row["Longitude"])
                                        try:
                                            r_nodes, dist_m, travel_min = calculate_route(agent.current_coords, h_coords)
                                        except Exception:
                                            # Fallback
                                            dist_km = haversine_distance(agent.current_coords, h_coords)
                                            travel_min = (dist_km / 40.0) * 60.0
                                            r_nodes = []
                                            
                                        expected_arr = current_time + travel_min
                                        
                                        # Attempt lock on the new hospital
                                        new_lock = lock_manager.claim_lock(h_name, agent.agent_id, expected_arr, current_time)
                                        if new_lock:
                                            # Successfully swapped lock!
                                            # Release the old lock
                                            if agent.lock_id:
                                                lock_manager.release_lock(agent.lock_id)
                                                
                                            # Update agent destination and routing
                                            agent.lock_id = new_lock.lock_id
                                            agent.dest_hospital = h_name
                                            agent.dest_coords = h_coords
                                            agent.route_nodes = r_nodes
                                            agent.route_coords = get_route_coords(r_nodes) if r_nodes else [agent.current_coords, h_coords]
                                            agent.total_distance = dist_m if r_nodes else dist_km * 1000
                                            agent.total_travel_time = travel_min
                                            agent.elapsed_travel_time = 0.0
                                            agent.lock_swaps += 1
                                            
                                            print(f"  [LOCK SWAP] {agent.agent_id} swapped to {h_name} at t={current_time:.1f} (ETA: {travel_min:.1f}m)")
                                            break
                                            
                    still_active.append(agent)
            
            elif agent.status in ("ADMITTED", "WAITING"):
                # Checked for admission during queue processing, or completed release.
                if agent.status == "ADMITTED" and current_time >= agent.release_time:
                    agent.status = "COMPLETED"
                    completed_agents.append(agent)
                else:
                    still_active.append(agent)
                    
        active_agents = still_active
        current_time += dt

    # Finalize any remaining active agents as they were cut off
    for agent in active_agents:
        if agent.status == "ADMITTED":
            agent.status = "COMPLETED"
            completed_agents.append(agent)
        elif agent.status == "WAITING":
            # Force metrics calculations for waiting agents using max simulation time
            agent.waiting_time = max_time - agent.arrival_time
            agent.time_to_admission = max_time - agent.call_arrival_time
            completed_agents.append(agent)
        elif agent.status == "TRAVELLING":
            # Driving was cut off
            agent.waiting_time = 0.0
            agent.time_to_admission = agent.total_travel_time
            completed_agents.append(agent)

    # Compile metrics
    times_to_admission = [a.time_to_admission for a in completed_agents]
    waiting_times = [a.waiting_time for a in completed_agents]
    
    mean_admission = np.mean(times_to_admission) if times_to_admission else 0.0
    p95_admission = np.percentile(times_to_admission, 95) if times_to_admission else 0.0
    mean_wait = np.mean(waiting_times) if waiting_times else 0.0
    
    total_swaps = sum(a.lock_swaps for a in completed_agents)
    
    # Per-hospital admissions distribution
    hospital_counts = {h_name: 0 for h_name in lock_manager.pools}
    for a in completed_agents:
        if a.dest_hospital in hospital_counts:
            hospital_counts[a.dest_hospital] += 1

    return {
        "mode": mode,
        "overcommit_events": overcommit_events,
        "mean_time_to_admission": mean_admission,
        "p95_time_to_admission": p95_admission,
        "mean_waiting_time": mean_wait,
        "lock_swaps": total_swaps,
        "hospital_distribution": hospital_counts,
        "agents": completed_agents
    }

def haversine_distance(coord1, coord2):
    """Fallback distance calculator."""
    R = 6371.0
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c
