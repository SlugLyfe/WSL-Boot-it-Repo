from statsbombpy import sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# Fetch WSL matches and events
wsl_1819_matches = sb.matches(competition_id=37, season_id=4)
wsl_1920_matches = sb.matches(competition_id=37, season_id=42)
wsl_2021_matches = sb.matches(competition_id=37, season_id=90)
wsl_total_matches = pd.concat([wsl_1819_matches, wsl_1920_matches, wsl_2021_matches], ignore_index=True)

events_total_df = []
for match_id in wsl_total_matches["match_id"]:
    match_events = sb.events(match_id=match_id)
    match_events["match_id"] = match_id
    events_total_df.append(match_events)
df = pd.concat(events_total_df, ignore_index=True)

# Function to transform coordinates
def transform_coords(location, pitch_length=105, pitch_width=68):
    if location:
        return (location[0] * pitch_length / 120, location[1] * pitch_width / 80)
    return (None, None)

pitch = Pitch(line_color='black', pitch_type='custom', pitch_length=105, pitch_width=68)

# Filter for moving actions (Passes and Carries)
move_df = df[df['type'].isin(['Pass', 'Carry'])].copy()
move_df['x'], move_df['y'] = zip(*move_df['location'].map(transform_coords))
move_df['end_x'], move_df['end_y'] = zip(*move_df['pass_end_location'].combine_first(move_df['carry_end_location']).map(transform_coords))

# Filter passes that end out of bounds
move_df.dropna(subset=['x', 'y', 'end_x', 'end_y'], inplace=True)
move_df = move_df[(move_df['end_x'] > 0) & (move_df['end_x'] < 105) & (move_df['end_y'] > 0) & (move_df['end_y'] < 68)]

# Exclude unsuccessful passes
move_df['is_possession_lost'] = move_df['pass_outcome'].isin(['Incomplete', 'Out of bounds'])

# Define bin edges
bin_x = np.linspace(0, 105, 17)  # 16 bins for x (17 edges)
bin_y = np.linspace(0, 68, 13)   # 12 bins for y (13 edges)

# Calculate start and end sectors
move_df['start_sector'] = (
    np.clip(np.digitize(move_df['x'], bins=bin_x, right=False) - 1, 0, len(bin_x) - 2) +
    np.clip(np.digitize(move_df['y'], bins=bin_y, right=False) - 1, 0, len(bin_y) - 2) * (len(bin_x) - 1)
)

move_df['end_sector'] = (
    np.clip(np.digitize(move_df['end_x'], bins=bin_x, right=False) - 1, 0, len(bin_x) - 2) +
    np.clip(np.digitize(move_df['end_y'], bins=bin_y, right=False) - 1, 0, len(bin_y) - 2) * (len(bin_x) - 1)
)

# Create histograms for moves, shots, goals, and possession loss
move = pitch.bin_statistic(move_df['x'], move_df['y'], statistic='count', bins=(16, 12))
possession_loss = pitch.bin_statistic(move_df[move_df['is_possession_lost']]['x'], move_df[move_df['is_possession_lost']]['y'], statistic='count', bins=(16, 12))
shot_df = df[df['type'] == 'Shot'].copy()
shot_df['x'], shot_df['y'] = zip(*shot_df['location'].map(transform_coords))
shot = pitch.bin_statistic(shot_df['x'], shot_df['y'], statistic='count', bins=(16, 12))
goal_df = shot_df[shot_df['shot_outcome'] == 'Goal']
goal = pitch.bin_statistic(goal_df['x'], goal_df['y'], statistic='count', bins=(16, 12))

# Calculate probabilities
goal_probability = np.nan_to_num(goal['statistic'] / (shot['statistic'] + 1e-6))  # Avoid divide by zero
move_probability = move['statistic'] / (move['statistic'] + shot['statistic'] + 1e-6)
shot_probability = shot['statistic'] / (move['statistic'] + shot['statistic'] + 1e-6)
possession_loss_probability = np.nan_to_num(possession_loss['statistic'] / (move['statistic'] + 1e-6))

# Transition matrices
transition_matrices = []
for start_sector in range(192):  # 16x12 = 192 zones
    start_actions = move_df[move_df['start_sector'] == start_sector]
    end_counts = start_actions['end_sector'].value_counts()
    T_matrix = np.zeros((12, 16))
    for end_sector, count in end_counts.items():
        end_x = end_sector % 16
        end_y = end_sector // 16
        if 0 <= end_x < 16 and 0 <= end_y < 12:
            T_matrix[end_y, end_x] = count
    if T_matrix.sum() > 0:
        T_matrix /= T_matrix.sum()
    transition_matrices.append(T_matrix)

# Convert transition matrices to a numpy array
transition_matrices_array = np.array(transition_matrices)

# Initialize xT matrix
xT = np.zeros((12, 16))

# Iteratively calculate xT for 10 moves
for iteration in range(10):
    # Direct payoff: Scoring directly from a shot
    shoot_expected_payoff = goal_probability * shot_probability

    # Indirect payoff: Moving the ball to another zone
    move_expected_payoff = (
        move_probability *
        np.sum(transition_matrices_array * xT, axis=(1, 2)).reshape(16, 12).T
    )

    # Combine payoffs and adjust for possession loss
    xT = (1 - possession_loss_probability) * (shoot_expected_payoff + move_expected_payoff)

# Normalize xT to represent probabilities (0 to 1)
xT = np.clip(xT, 0, 1) 

# Plot the final xT heatmap
xT_statistic = pitch.bin_statistic([], [], bins=(16, 12))
xT_statistic['statistic'] = xT  # Assign xT matrix to statistic field

fig, ax = pitch.draw(figsize=(12, 8))
pcm = pitch.heatmap(xT_statistic, cmap='Oranges', edgecolor='grey', ax=ax)
pitch.label_heatmap(xT_statistic, color='black', fontsize=8, ax=ax, str_format="{:.2f}")
plt.colorbar(pcm, ax=ax, fraction=0.03, pad=0.04)
plt.title('Expected Threat (xT) Heatmap after 10 moves', fontsize=16)
plt.show()
np.save("xT_map.npy", xT)
print("xT map saved to xT_map.npy.")
