import os
import matplotlib.pyplot as plt

print("[VISUALIZATION] Initializing academic chart generator...")

# Exact data extracted from your successful 1.90s terminal execution logs
hospitals = [
    'Ramesh Hospital', 
    'Vijaya Super Speciality', 
    'Andhra Hospital', 
    'Care Hospital', 
    'Government General', 
    'NRI Hospital', 
    'Manipal Hospital'
]
transit_times_mins = [4.4, 4.4, 11.8, 11.8, 14.8, 15.0, 15.0]

# Configure a clean, professional plot canvas
fig, ax = plt.subplots(figsize=(10, 5.5))

# Define an executive palette: Green for optimal targets, slate gray for non-optimal
colors = ['#2ecc71' if x == min(transit_times_mins) else '#34495e' for x in transit_times_mins]

# Generate the horizontal bar chart
bars = ax.barh(hospitals, transit_times_mins, color=colors, edgecolor='#2c3e50', height=0.55, linewidth=1.2)

# Design clean academic borders and gridlines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#7f8c8d')
ax.spines['bottom'].set_color('#7f8c8d')
ax.xaxis.grid(True, linestyle='--', alpha=0.6, color='#bdc3c7')
ax.set_axisbelow(True)

# Add explicit chart text descriptions
ax.set_xlabel('Predicted Ambulance Transit Window (Minutes)', fontsize=11, fontweight='bold', labelpad=10, color='#2c3e50')
ax.set_title('Dynamic Emergency Route Optimization Analysis\nSpatial Environment: Vijayawada, India (Weather: CLEAR)', 
             fontsize=13, fontweight='bold', pad=20, color='#2c3e50')

# Invert Y-axis so the fastest recommended option rests gracefully at the top
ax.invert_yaxis()

# Append exact data metrics values right next to the bar elements
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, f'{width:.1f} mins', 
            va='center', ha='left', fontsize=10, fontweight='bold', color='#2c3e50')

# Establish safe horizontal boundary buffer
ax.set_xlim(0, 18)
plt.tight_layout()

# Save publication-grade asset directly to workspace directory
output_img = "routing_performance_chart.png"
plt.savefig(output_img, dpi=300)

print("="*65)
print(f"[SUCCESS] Academic chart successfully saved as: {output_img}")
print("You can now open this image file to insert directly into your report!")
print("="*65)
