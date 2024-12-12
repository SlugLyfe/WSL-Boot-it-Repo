import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pickle
import pandas as pd

# Define bins for the pitch
x_bins = np.linspace(0, 105, 17)  # 16 bins for x
y_bins = np.linspace(0, 68, 13)  # 12 bins for y

def calculate_goal_probability(events_df, team_name):
    # Filter backward passes in the team's first/build-up third
    backward_passes = events_df[
        (events_df["team"] == team_name) &
        (events_df["type"] == "Pass") &
        (events_df["pass_angle"].apply(lambda angle: angle < -np.pi / 2 or angle > np.pi / 2)) &
        (events_df["location"].apply(lambda loc: isinstance(loc, list) and loc[0] <= 40))
    ]

    # Initialize scoring probability matrix
    scoring_matrix = np.zeros((12, 16))  # 12 rows (y), 16 columns (x)

    for _, pass_event in backward_passes.iterrows():
        # Determine the bin where the pass was received
        end_location = pass_event["pass_end_location"]
        if not isinstance(end_location, list):  # Skip invalid locations
            continue
        x_bin = min(np.digitize(end_location[0], x_bins) - 1, scoring_matrix.shape[1] - 1)
        y_bin = min(np.digitize(end_location[1], y_bins) - 1, scoring_matrix.shape[0] - 1)

        # Filter possession chain events for this backward pass
        possession_chain = events_df[
            (events_df["possession"] == pass_event["possession"]) &
            (events_df["index"] >= pass_event["index"]) &
            (events_df["index"] <= pass_event["index"] + 10)
        ]

        # Check if the team scored a goal within 10 moves
        if (
            not possession_chain.empty and
            (possession_chain["type"] == "Shot").any() and
            (possession_chain[(possession_chain["type"] == "Shot") & 
                              (possession_chain["shot_outcome"] == "Goal")]["team"] == team_name).any()
        ):
            scoring_matrix[y_bin, x_bin] += 1

    # Normalize by total backward passes received in each bin
    backward_passes_received = np.zeros((12, 16))
    for _, pass_event in backward_passes.iterrows():
        end_location = pass_event["pass_end_location"]
        if not isinstance(end_location, list):  # Skip invalid locations
            continue
        x_bin = min(np.digitize(end_location[0], x_bins) - 1, backward_passes_received.shape[1] - 1)
        y_bin = min(np.digitize(end_location[1], y_bins) - 1, backward_passes_received.shape[0] - 1)
        backward_passes_received[y_bin, x_bin] += 1

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        scoring_probability_matrix = np.nan_to_num(
            (scoring_matrix / backward_passes_received) * 100
        )  # Convert to percentage

    return scoring_probability_matrix

# Load the processed dataframe
events_with_xt = pd.read_pickle("events_with_xt.pkl")

# Create matrices for all teams
teams = events_with_xt["team"].unique()
team_probability_matrices = {}

for team in teams:
    print(f"Processing team: {team}")
    team_matrix = calculate_goal_probability(events_with_xt, team)
    team_probability_matrices[team] = team_matrix

# Save the results to pickle files
with open("backward_pass_probability_matrices.pkl", "wb") as f:
    pickle.dump(team_probability_matrices, f)

print("Backward pass probability matrices saved to backward_pass_probability_matrices.pkl.")

# Load the matrices from the pickle file
try:
    with open("backward_pass_probability_matrices.pkl", "rb") as f:
        team_probability_matrices = pickle.load(f)
except FileNotFoundError:
    print("Backward pass probability matrices file not found. Ensure you have the correct path.")
    team_probability_matrices = {}

# Define the soccer pitch
pitch = Pitch(pitch_type='statsbomb', pitch_length=105, pitch_width=68,
              line_color='black', line_zorder=2)

# Choose a team for visualization
team_to_plot = "Arsenal WFC"  

if team_to_plot in team_probability_matrices:
    matrix = team_probability_matrices[team_to_plot]

    # Define the x and y bin edges (grid edges)
    x_bins = np.linspace(0, 105, matrix.shape[1] + 1)  # x-axis bin edges
    y_bins = np.linspace(0, 68, matrix.shape[0] + 1)  # y-axis bin edges

    # Define the x and y bin centers (grid centers)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    y_centers = (y_bins[:-1] + y_bins[1:]) / 2

    # Plot the soccer pitch
    fig, ax = pitch.draw(figsize=(12, 8))

    # Plot the heatmap using pcolormesh
    pcm = ax.pcolormesh(x_bins, y_bins, matrix, cmap="Oranges", shading="flat", alpha=0.9)

    # Add numerical labels to each bin
    for i, y in enumerate(y_centers):
        for j, x in enumerate(x_centers):
            value = matrix[i, j]
            if not np.isnan(value) and value > 0:
                ax.text(
                    x, y,
                    f"{value:.2f}%",  # Percentage format
                    color='black', ha='center', va='center', fontsize=8, zorder=3
                )

    # Add a color bar
    cbar = plt.colorbar(pcm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.05)
    cbar.set_label('Goal Probability (%)', fontsize=12)

    # Add a title
    plt.title(f"Goal Probability Matrix for Backward Passes - {team_to_plot}", fontsize=16)

    plt.tight_layout()
    plt.show()
else:
    print(f"Team '{team_to_plot}' not found in the dataset.")
