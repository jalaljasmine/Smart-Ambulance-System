# 🚑 Smart Ambulance Routing & Traffic Optimization System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Flask-3.1-green.svg)](https://flask.palletsprojects.com/)
[![ML](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org/)
[![Routing](https://img.shields.io/badge/OSMnx-NetworkX-purple.svg)](https://osmnx.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

An end-to-end AI-powered emergency response and route optimization platform built for Expecially Vijayawada. The system predicts **Accident Severity**, recommends the **Best Hospital** based on capacity, estimates **Traffic Congestion**, and calculates the **Fastest Ambulance Route** using **Dijkstra's Algorithm** on real OpenStreetMap data for **Vijayawada, Andhra Pradesh, India**.

---

## 📌 Problem Statement

During emergency situations (e.g., road accidents), traditional dispatching systems encounter severe bottlenecks:
- Delayed assessment of victim injury severity leading to inappropriate hospital selection.
- Static GPS routing ignoring live traffic congestion and road bottlenecks.
- Lack of real-time visibility into hospital ICU bed capacity and emergency staff.

Every minute lost in transit reduces patient survival rates ("The Golden Hour").

---

## 💡 Solution

The **Smart Ambulance System** solves these challenges by combining Machine Learning, Multi-Criteria Decision Making, and Graph Theory into a unified, 4-stage pipeline:

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│  User Inputs    │ ──► │  Accident Severity   │ ──► │ Hospital               │
│  (Accident Form)│     │  (Random Forest ML)  │     │ Recommendation Engine  │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
                                                                 │
                                                                 ▼
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│ Results & Map   │ ◄── │ Shortest Route       │ ◄── │ Traffic Volume          │
│ (Flask Dashboard)│     │ (Dijkstra Algorithm) │     │ (Random Forest ML)     │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
```

---

## ✨ Key Features

- **ML Accident Severity Prediction**: Classifies injury severity (`Slight Injury`, `Serious Injury`, `Fatal Injury`) using a trained Random Forest model.
- ** Smart Hospital Scoring**: Ranks hospitals using a weighted score combining ICU Beds, Emergency Staff, Trauma Center availability, and accident severity.
- **Traffic Congestion Forecasting**: Predicts vehicle volume per junction/hour using Random Forest Regression to dynamically adjust ambulance speed.
- **Route Optimization (Dijkstra)**: Calculates shortest node-path using Dijkstra's algorithm on a real 11MB GraphML road network downloaded via OSMnx.
- ** Interactive Web Dashboard**: Built with Flask and Folium for interactive dark-mode visualization of maps, route metrics, and hospital rankings.
- **Console / CLI Support**: Includes a zero-dependency CLI mode with `--demo` flag for fast command-line execution.

---

## 🛠️ Tech Stack & Libraries

| Domain | Technologies |
|--------|--------------|
| **Language** | Python 3.10+ |
| **Web Framework** | Flask, Jinja2, HTML5, CSS3 (Vanilla Glassmorphism) |
| **Machine Learning** | Scikit-Learn, Joblib, Pandas, NumPy |
| **Graph & Routing** | OSMnx, NetworkX, GeoPandas, Shapely |
| **Visualization** | Folium, Matplotlib, Branca |

---

## 📊 Machine Learning Models & Datasets

### 1. Accident Severity Model (`models/accident_model.pkl`)
- **Algorithm**: Random Forest Classifier (`n_estimators=100`)
- **Dataset**: `datasets/accidents.csv` (12,318 rows, 32 columns)
- **Features (31)**: Time, Day of week, Weather, Light conditions, Road surface, Collision type, Cause of accident, Vehicles involved, Casualties, etc.
- **Target**: `Accident_severity` (0: Fatal, 1: Serious, 2: Slight)

### 2. Traffic Prediction Model (`models/traffic_model.pkl`)
- **Algorithm**: Random Forest Regressor (`n_estimators=100`)
- **Dataset**: `datasets/traffic.csv` (48,122 rows)
- **Features (5)**: `[Hour, Day, Month, Weekday, Junction]`
- **Target**: `Vehicles` (Vehicle count per hour)

### 3. Road Network Graph (`datasets/vijayawada.graphml`)
- **Coverage**: Vijayawada, Andhra Pradesh, India
- **Source**: OpenStreetMap via OSMnx
- **Nodes/Edges**: Full drivable street network

---

## 📂 Project Structure

```
smart-ambulance-system/
│
├── config.py                      # Centralized configuration (paths, defaults, labels)
├── main.py                        # Console / CLI entry point (--demo supported)
├── requirements.txt               # Full Python dependency list
├── README.md                      # Project documentation
│
├── dashboard/
│   └── app.py                     # Flask Web Application (routes & Folium map generator)
│
├── models/
│   ├── accident_model.pkl         # Trained Random Forest Severity Classifier (28 MB)
│   ├── traffic_model.pkl          # Trained Random Forest Traffic Regressor (319 MB)
│   ├── predict_severity.py        # Severity prediction & LabelEncoder logic
│   ├── hospital_score.py          # Hospital scoring & ranking logic
│   ├── accident-model-pkl.ipynb   # Notebook: Accident Severity Training
│   └── traffic-model-pkl.ipynb    # Notebook: Traffic Model Training
│
├── routing/
│   ├── dijkstra.py                # GraphML load & Dijkstra shortest path calculator
│   ├── check_traffic.py           # Traffic model inference wrapper
│   ├── download_roads.py          # Utility script to download OSMnx road graph
│   └── check_roads.py             # Utility script to inspect roads dataset
│
├── templates/
│   ├── index.html                 # Web dashboard input form
│   └── result.html                # Web dashboard results page + embedded map
│
├── datasets/
│   ├── accidents.csv              # 12K road accident dataset
│   ├── hospitals.csv              # Vijayawada hospitals (beds, trauma center, lat/lon)
│   ├── traffic.csv                # 48K traffic dataset
│   ├── roads.csv                  # Road network edge data
│   └── vijayawada.graphml         # Road network graph (OSMnx GraphML format)
│
└── notebooks/                     # Exploratory Data Analysis & Jupyter Notebooks
```

---

## ⚡ Quick Start & Installation

### 1. Prerequisites
- Python 3.10+ installed on your system.

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/smart-ambulance-system.git
cd smart-ambulance-system

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Option A: Web Dashboard (Recommended for Demos)
Run the Flask server:
```bash
python dashboard/app.py
```
Open your browser and navigate to:
👉 **`http://localhost:5000`**

Fill in the accident details and click **"Run Smart Ambulance Pipeline"** to view:
- Predicted Injury Severity
- Recommended Hospital & Score
- Predicted Traffic Volume
- Shortest Route (Distance in km & ETA in minutes)
- Interactive Map & Ranked Hospital Table

---

### Option B: Command Line Interface (CLI)

#### Run with preset demo scenario:
```bash
python main.py --demo
```

#### Run interactively:
```bash
python main.py
```

**Sample CLI Output:**
```text
==================================================
   [*] SMART AMBULANCE ROUTING SYSTEM
       Vijayawada, Andhra Pradesh
==================================================

[DEMO] Running with preset accident scenario

[1] Predicting Accident Severity...
    Severity: Slight Injury

[2] Recommending Best Hospital...
    Hospital    : Government General Hospital
    ICU Beds    : 10  | Trauma Center: Yes

[3] Predicting Traffic Volume...
    Predicted Vehicles/Hour: 30.57

[4] Calculating Shortest Route (Dijkstra)...
  [Router] Loading road network from disk...

==================================================
             FINAL RESULTS
==================================================
  Accident Severity   : Slight Injury
  Recommended Hospital: Government General Hospital
  ICU Beds            : 10
  Trauma Center       : Yes
  Traffic (veh/hr)    : 30.57
  Route Distance      : 2.45 km
  Estimated ETA       : 7.34 minutes
  Route Nodes         : 34
==================================================
[OK] Pipeline completed successfully.


## 🔮 Future Scope

- **📍 Live GPS Integration**: Real-time ambulance tracking via WebSockets.
- **🚦 Smart Traffic Signal Preemption**: Automatically turn traffic signals green along the ambulance path.
- **🌦️ Weather API & Live Traffic APIs**: Real-time TomTom / Google Traffic API integration.
- **📱 Mobile App**: Flutter/React Native application for paramedics.

---

## 📄 License

<<<<<<< HEAD
This project is developed for educational and research purposes.
..
=======
This project is developed for educational, research, and B.Tech capstone purposes under the MIT License.
>>>>>>> 921f84e (Updated Smart Ambulance Routing Project)
