import matplotlib.pyplot as plt
from mplsoccer import Pitch
import numpy as np

# Define scoring probabilities
scoring_probabilities = {
    "60-75m": 0.0049,  # 0.48%
    "75-90m": 0.0054,  # 0.54%
    "90-105m": 0.0058,  # 0.58%
    "105-120m": 0.0047  # 0.47%
}

# Map regions to probabilities
x_regions = [60, 75, 90, 105, 120]  # Boundaries for bins
probabilities = [scoring_probabilities["60-75m"], scoring_probabilities["75-90m"], 
                 scoring_probabilities["90-105m"], scoring_probabilities["105-120m"]]

# Create a soccer pitch
pitch = Pitch(line_color="black", pitch_length=120, pitch_width=80)
fig, ax = pitch.draw(figsize=(14, 8))

# Add heatmap bins manually
for i in range(len(probabilities)):
    ax.fill(
        [x_regions[i], x_regions[i + 1], x_regions[i + 1], x_regions[i]],
        [0, 0, 80, 80],
        color=plt.cm.Oranges(probabilities[i] / max(probabilities)),
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5
    )

# Add a color bar
sm = plt.cm.ScalarMappable(cmap="Oranges", norm=plt.Normalize(0, max(probabilities)))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.1)
cbar.set_label("Scoring Probability (%)", fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)

# Add labels for the regions
for i in range(len(probabilities)):
    mid_x = (x_regions[i] + x_regions[i + 1]) / 2
    ax.text(
        mid_x, 40, f"{probabilities[i] * 100:.2f}%", 
        color="black", fontsize=14, weight="bold",
        ha="center", va="center",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3", alpha=0.8)
    )

# Enhance title and layout
plt.title(
    "Scoring Probability by Throw-In Region", 
    fontsize=18, weight="bold", pad=20
)
plt.tight_layout()
plt.show()


# Load xT data (assuming it's saved in a numpy file)
xT = np.load("xT_map.npy")  # Replace with the correct file path if different

# Check the dimensions of xT
print(f"xT shape: {xT.shape}")

# Create a soccer pitch
pitch = Pitch(pitch_type='statsbomb', pitch_length=120, pitch_width=80,
              line_color='black', line_zorder=2)

# Define the x and y bin centers (grid centers)
x_bins = np.linspace(0, 120, xT.shape[1] + 1)  # x-axis bin edges
y_bins = np.linspace(0, 80, xT.shape[0] + 1)  # y-axis bin edges
x_centers = (x_bins[:-1] + x_bins[1:]) / 2  # x-axis bin centers
y_centers = (y_bins[:-1] + y_bins[1:]) / 2  # y-axis bin centers

# Plot the pitch and heatmap
fig, ax = pitch.draw(figsize=(12, 8))
heatmap = ax.pcolormesh(x_bins, y_bins, xT, cmap='Oranges', edgecolors='grey', shading='flat')

# Add numerical labels to each bin
for i, y in enumerate(y_centers):  # Iterate through y-axis bins
    for j, x in enumerate(x_centers):  # Iterate through x-axis bins
        ax.text(
            x, y,  # Position at the center of each bin
            f"{xT[i, j]:.2f}",  # Value to display
            color='blue', ha='center', va='center', fontsize=8, zorder=3
        )

# Add a color bar
cbar = plt.colorbar(heatmap, ax=ax, orientation='horizontal', fraction=0.05, pad=0.05)
cbar.set_label('xT Value', fontsize=12)

# Add a title
plt.title('Expected Threat (xT) Heatmap with Soccer Pitch', fontsize=16)

plt.tight_layout()
plt.show()