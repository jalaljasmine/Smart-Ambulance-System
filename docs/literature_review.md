# 📚 Literature Review & Theoretical Background
## Smart Ambulance Routing & Traffic Optimization System

---

## 1. Introduction & Golden Hour Principle

In emergency medical services (EMS), the **"Golden Hour"** refers to the first 60 minutes following a traumatic injury. Survival rates drop dramatically if emergency care and hospital transfer are delayed past this window. Efficient ambulance dispatching, severity-based triage, and traffic-aware route optimization are essential components of smart city emergency infrastructure.

---

## 2. Key Research Papers & References

1. **Shortest Path Algorithms in Emergency Logistics**
   - *Dijkstra, E. W. (1959)*: "A note on two problems in connexion with graphs". *Numerische Mathematik*, 1(1), 269–271.
   - *Application*: Used for computing exact shortest paths on road network graphs ($G = (V, E)$).

2. **Machine Learning for Road Accident Severity Classification**
   - *Chong, M. M., et al. (2005)*: "Traffic accident analysis using machine learning paradigms". *International Journal of Simulation*, 6(3), 12–18.
   - *Application*: Demonstrates Random Forest and Decision Tree classifiers outperforming traditional logistic regression for accident severity triage.

3. **Traffic Congestion Forecasting using Random Forest Regression**
   - *Vlahogianni, E. I., et al. (2014)*: "Short-term traffic forecasting: Where we are and where we're going". *Transportation Research Part C: Emerging Technologies*, 43, 3–19.
   - *Application*: Short-term vehicle volume prediction at junctions based on temporal features (hour, weekday, month).

4. **Multi-Criteria Hospital Selection in Smart Cities**
   - *Talaei, H., et al. (2020)*: "A GIS-based multi-criteria decision-making approach for hospital site selection and emergency routing". *Journal of Healthcare Engineering*.
   - *Application*: Scoring hospitals based on available ICU beds, trauma center certification, and medical staff capacity.

---

## 3. Algorithm Selection Justification

### Why Dijkstra's Algorithm over A* or Bellman-Ford?
- **Exact Shortest Path**: Dijkstra guarantees the global optimal path on non-negative weighted graphs (road length & travel time weights).
- **Single-Source Efficiency**: With pre-built GraphML network structures, Dijkstra executes in $\mathcal{O}((V + E) \log V)$ time using priority queues, achieving real-time performance (< 1 second) for city-level graphs like Vijayawada (~10,000 nodes).
