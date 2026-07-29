# pyrefly: ignore [missing-import]
import osmnx as ox

place = "Vijayawada, Andhra Pradesh, India"

print("Downloading road network...")

G = ox.graph_from_place(place, network_type="drive")

# Save graph for future use
ox.save_graphml(G, "datasets/vijayawada.graphml")

print("Graph saved successfully!")

# Optional: also save roads.csv
nodes, edges = ox.graph_to_gdfs(G)
edges.to_csv("datasets/roads.csv", index=False)

print("roads.csv created successfully!")