# 🚑 Smart Ambulance Routing & Traffic Optimization System

An AI-powered Smart City project that helps emergency services recommend the best hospital and the fastest ambulance route by combining Machine Learning, Traffic Prediction, Hospital Recommendation, and Route Optimization.

---

## 📌 Project Overview

During a road accident, every minute is critical. This project predicts accident severity, recommends a suitable hospital, estimates traffic conditions, and finds the shortest route for the ambulance using Dijkstra's algorithm.

The system is designed to reduce ambulance response time and support better emergency decision-making.

---

## ✨ Features

- Accident Severity Prediction using Machine Learning
- Hospital Recommendation based on severity
- Traffic Prediction using ML model
- Shortest Route Calculation using Dijkstra Algorithm
- Road Network generated using OpenStreetMap (OSMnx)
- Estimated Travel Time Calculation
- Modular Python Project Structure
- Ready for Future GPS & Weather Integration

---

## 🛠 Tech Stack

- Python
- Pandas
- Scikit-learn
- Joblib
- NetworkX
- OSMnx
- OpenStreetMap
- Jupyter Notebook
- VS Code

---

## 📂 Project Structure

```
smart-ambulance-system/

│
├── datasets/
│   ├── accidents.csv
│   ├── hospitals.csv
│   ├── traffic.csv
│   ├── roads.csv
│   ├── weather.csv
│   └── vijayawada.graphml
│
├── models/
│   ├── accident_model.pkl
│   ├── traffic_model.pkl
│   ├── predict_severity.py
│   └── hospital_score.py
│
├── routing/
│   ├── check_traffic.py
│   ├── download_roads.py
│   ├── dijkstra.py
│   └── check_roads.py
│
├── notebooks/
│   ├── accident_severity.ipynb
│   ├── traffic_prediction.ipynb
│   ├── hospital_recommendation.ipynb
│   ├── data_cleaning.ipynb
│   └── eda.ipynb
│
├── docs/
├── dashboard/
├── main.py
└── README.md
```

---

## ⚙️ Workflow

```
Accident Occurs
        │
        ▼
Accident Details
        │
        ▼
Accident Severity Prediction (ML)
        │
        ▼
Hospital Recommendation
        │
        ▼
Traffic Prediction (ML)
        │
        ▼
Shortest Route (Dijkstra)
        │
        ▼
Estimated Travel Time
        │
        ▼
Recommended Hospital & Route
```

---

## 📊 Machine Learning Models

### Accident Severity Model

Predicts:

- Slight Injury
- Serious Injury
- Fatal Injury

---

### Traffic Prediction Model

Predicts traffic congestion using historical traffic data.

---

## 🏥 Hospital Recommendation

Hospitals are ranked using:

- ICU Beds
- Trauma Center Availability
- Emergency Staff
- Accident Severity

The hospital with the highest score is recommended.

---

## 🛣 Route Optimization

Uses:

- OpenStreetMap
- OSMnx
- NetworkX
- Dijkstra Algorithm

Outputs:

- Shortest Route
- Distance
- Estimated Travel Time

---

## ▶️ Run the Project

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python main.py
```

---

## 📌 Current Output

```
Predicted Severity : Serious Injury

Predicted Traffic : 9.82

Recommended Hospital :
Government General Hospital

Distance : 3.25 km

Estimated Time : 4.88 minutes
```

---

## 🔒 Reservation-Lock Coordination Layer & Simulation

To address the hospital over-commitment issue where multiple ambulances concurrently target the same hospital based on stale bed counts, we introduced a **Reservation-Lock Coordination Layer**.

### Features:
- **Lock Manager (`coordination/lock_manager.py`)**: Tracks in-memory bed resource pools and manages atomic `ReservationLock` claims (status: `HELD`, `RELEASED`, `CONVERTED`, `EXPIRED`) with a time-to-live (TTL = expected arrival + buffer).
- **Lock-Aware Dispatch Flow (`models/hospital_score.py`)**: Attempts to claim a lock on the top-scored hospital. If full, it falls back down the sorted candidate leaderboard until it secures a lock.
- **Multi-Ambulance Simulator (`simulation/multi_ambulance_sim.py`)**: Runs N ambulances on the road graph with random arrival times, severities, travel times, and dynamic route re-evaluations (swapping locks if a better hospital opens up).
- **Offline Cache Optimization**: Transparently monkeypatches OSMnx graph-loading in the simulation to run hundreds of times faster.

### Run the Simulation Comparison:
Run the comparison script to simulate both `BASELINE` (no reservation locks) and `LOCKED` modes on the same sequence of calls:

```bash
python -m results.compare_conditions --ambulances 15 --bed-scale 0.05 --seed 42
```

This prints a metrics comparison table (over-commitment count, average waiting times, time-to-admission, lock swaps) and saves a comparative plot at `results/condition_comparison.png`.

---

## 🚀 Future Improvements

- Live GPS Location
- Live Traffic API
- Weather API Integration
- Hospital Bed Availability API
- Google Maps Integration
- Real-time Ambulance Tracking
- Flask/FastAPI Backend
- React Dashboard
- Multi-Ambulance Coordination
- Smart Traffic Signal Control
- IoT Sensor Integration

---
## 📄 License

This project is developed for educational and research purposes.
..