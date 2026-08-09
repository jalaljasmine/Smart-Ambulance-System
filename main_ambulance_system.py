import joblib
import networkx as nx
import pandas as pd
import numpy as np
import time

print("[SYSTEM] Initializing Master Weather-Aware Ambulance Framework...")
start_init = time.time()

# 1. Load All Core Assets Safely from Workspace
G = nx.read_graphml("datasets/vijayawada.graphml")
hospitals_df = pd.read_csv("datasets/hospitals.csv")
traffic_model = joblib.load("models/traffic_model.pkl")
weather_registry = pd.read_csv("datasets/weather_clean.csv")

try:
    accident_model = joblib.load("models/accident_model.pkl")
    print("[SYSTEM] Accident Model integrated successfully.")
except:
    accident_model = None
    print("[WARNING] Accident binary loaded as structural fallback.")

# 2. Vectorized Rapid Routing Engine Layer (With Climate Friction Overlay)
def generate_predictive_network_fast(graph, hour, day, month, weekday, current_weather="Clear"):
    current_weather = str(current_weather).strip()
    print(f"[ENGINE] Preparing batch feature matrix for weather state: [{current_weather}]...")
    live_graph = graph.copy()
    
    edges_list = list(live_graph.edges(keys=True, data=True))
    feature_rows = []
    
    for u, v, k, data in edges_list:
        highway_type = data.get('highway', 'residential')
        junction_proxy = 1 if highway_type in ['primary', 'trunk'] else 2
        feature_rows.append([hour, day, month, weekday, junction_proxy])
        
    features_df = pd.DataFrame(feature_rows, columns=["Hour", "Day", "Month", "Weekday", "Junction"])
    all_predictions = traffic_model.predict(features_df)
    
    print("[ENGINE] Injecting dynamic friction and climate costs into spatial matrices...")
    for idx, (u, v, k, data) in enumerate(edges_list):
        length = float(data.get('length', 100))
        
        # --- ROBUST MAXSPEED CLEANING LOGIC ---
        raw_speed = data.get('maxspeed', 40)
        if isinstance(raw_speed, list):
            valid_speeds = [int(x) for x in raw_speed if str(x).isdigit()]
            maxspeed = max(valid_speeds) if valid_speeds else 40
        elif isinstance(raw_speed, str) and '[' in raw_speed:
            try:
                clean_list = eval(raw_speed)
                valid_speeds = [int(x) for x in clean_list if str(x).isdigit()]
                maxspeed = max(valid_speeds) if valid_speeds else 40
            except:
                maxspeed = 40
        else:
            try:
                maxspeed = float(raw_speed)
            except:
                maxspeed = 40
        
        # --- ROBUST LANES CLEANING LOGIC ---
        raw_lanes = data.get('lanes', 2)
        if isinstance(raw_lanes, list):
            valid_lanes = [int(x) for x in raw_lanes if str(x).isdigit()]
            lanes = max(valid_lanes) if valid_lanes else 2
        elif isinstance(raw_lanes, str) and '[' in raw_lanes:
            try:
                clean_list = eval(raw_lanes)
                valid_lanes = [int(x) for x in clean_list if str(x).isdigit()]
                lanes = max(valid_lanes) if valid_lanes else 2
            except:
                lanes = 2
        else:
            try:
                lanes = int(float(raw_lanes))
            except:
                lanes = 2
                
        if lanes <= 0:
            lanes = 2

        base_time = length / (maxspeed * 1000 / 3600)
        
        weather_multiplier = 1.0
        if current_weather == "Rainy":
            weather_multiplier = 1.30
            if highway_type in ['residential', 'tertiary']:
                weather_multiplier = 1.45
        elif current_weather == "Foggy":
            weather_multiplier = 1.40
            
        predicted_vehicles = all_predictions[idx]
        congestion_multiplier = 1 + (predicted_vehicles / (lanes * 12))
        
        live_graph[u][v][k]['dynamic_cost'] = base_time * congestion_multiplier * weather_multiplier
        
    return live_graph

# 3. Medical Priority Destination Recommender (Multi-Output Matrix Generation)
def recommend_optimal_hospital(hospitals_tab, live_network, ambulance_current_node, patient_severity="High"):
    candidate_hospitals = hospitals_tab.copy()
    
    if patient_severity == "High":
        candidate_hospitals = candidate_hospitals[candidate_hospitals['Trauma_Center'] == 'Yes']
    candidate_hospitals = candidate_hospitals[candidate_hospitals['ICU_Beds'] > 0]
    
    all_results_list = []
    all_graph_nodes = list(live_network.nodes())
    
    # Geographically-spread mapping linking real hospital metrics to spread-out spatial node IDs
    hospital_node_mapping = {
        'Government General Hospital': "571470125",
        'NRI Hospital': "660699662",
        'Andhra Hospital': "660699683",
        'Ramesh Hospital': "1880441490",
        'Manipal Hospital Vijayawada': "660700031",
        'Care Hospital Vijayawada': "708920890",
        'Vijaya Super Speciality Hospital': "1880441510",
    }
    
    print("\n[RECOMMENDER] Matrix Lookup Active. Commencing complete multi-path computation routing...")
    
    for idx, row in candidate_hospitals.iterrows():
        h_name = row['Hospital']
        mock_target_node = hospital_node_mapping.get(h_name, all_graph_nodes[0])
        
        if mock_target_node not in all_graph_nodes:
            mock_target_node = all_graph_nodes[idx % len(all_graph_nodes)]
            
        try:
            route = nx.shortest_path(live_network, source=ambulance_current_node, target=mock_target_node, weight='dynamic_cost')
            
            total_time = 0
            for i in range(len(route)-1):
                edge_dict = live_network.get_edge_data(route[i], route[i+1])
                # FIX: Extract the integer item by position zero index [0] to keep it hashable
                first_key = list(edge_dict.keys())[0]
                total_time += edge_dict[first_key]['dynamic_cost']
                
            all_results_list.append({
                'Hospital': row['Hospital'],
                'Beds': row['ICU_Beds'],
                'Trauma': row['Trauma_Center'],
                'Time_Sec': total_time,
                'Time_Min': total_time / 60,
                'Nodes_Passed': len(route),
                'Path': route
            })
                
        except nx.NetworkXNoPath:
            continue
            
    leaderboard = pd.DataFrame(all_results_list).sort_values(by='Time_Sec').reset_index(drop=True)
    return leaderboard

# 4. Master Lookup & Execution Block
target_simulation_timestamp = "2015-11-01 17:00:00"

print(f"\n[SYSTEM] Querying environmental register for timestamp: {target_simulation_timestamp}")
matched_weather_row = weather_registry[weather_registry['DateTime'] == target_simulation_timestamp]

if not matched_weather_row.empty:
    # FIX: Extract the text string using .iloc[0] to drop the string array object formatting completely
    live_weather_condition = str(matched_weather_row['Weather_State'].iloc[0]).strip()
else:
    live_weather_condition = "Clear"

ts = pd.to_datetime(target_simulation_timestamp)
hour_val = ts.hour
day_val = ts.day
month_val = ts.month
weekday_val = ts.weekday()

all_nodes_list = list(G.nodes())

ambulance_start = "660700040" if "660700040" in all_nodes_list else all_nodes_list

# Generate routing costs layer
live_city_traffic = generate_predictive_network_fast(
    G, hour=hour_val, day=day_val, month=month_val, weekday=weekday_val, current_weather=live_weather_condition
)

# Fetch sorted data matrix layout
results_dataframe = recommend_optimal_hospital(
    hospitals_df, live_city_traffic, ambulance_current_node=ambulance_start, patient_severity="High"
)

# 5. Render Final Multi-Output Console Dashboard Layout
print("\n" + "="*85)
print(f"               AMBULANCE DISPATCH CONTROLLER - MULTI-OUTPUT ROUTING MATRIX             ")
print("="*85)
print(f"SIMULATION TIMESTAMP    : {target_simulation_timestamp}")
print(f"ENVIRONMENT CLIMATE DATA: {live_weather_condition.upper()}")
print(f"TOTAL OPTIONS VALIDATED : {len(results_dataframe)} Qualified Emergency Operations centers\n")

print(results_dataframe[['Hospital', 'Beds', 'Trauma', 'Time_Min', 'Nodes_Passed']].to_string(index=True))

print("-"*85)
best_option = results_dataframe.iloc[0]
print(f"CRITICAL OPTIMIZATION SUMMARY TARGET ACTION:")
print(f" -> Dispatch Ambulance to: {best_option['Hospital']}")
print(f" -> Minimum Transit Window  : {best_option['Time_Sec']:.1f} Seconds (~{best_option['Time_Min']:.1f} Minutes)")
print(f" -> Traversed Node String   : {len(best_option['Path'])} map intersections managed")
print(f" -> System Process Profiler : {time.time() - start_init:.2f} seconds engine evaluation benchmark")
print("="*85)
