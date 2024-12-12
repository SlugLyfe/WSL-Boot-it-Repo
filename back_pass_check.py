import numpy as np
import pandas as pd
import pickle

# Load required data
with open("backward_pass_probability_matrices.pkl", "rb") as f:
    backward_pass_matrices = pickle.load(f)

with open("defending_throw_in_probability_matrices.pkl", "rb") as f:
    throw_in_matrices = pickle.load(f)

with open("events_with_xt.pkl", "rb") as f:
    events_df = pickle.load(f)

with open("scoring_probabilities.pkl", "rb") as f:
    scoring_probabilities = pickle.load(f)

# Define pitch dimensions and bins
x_bins = np.linspace(0, 105, 17)  # 16 bins for x
y_bins = np.linspace(0, 68, 13)  # 12 bins for y

def find_bin(coord, bins):
    """Helper function to find the bin index for a coordinate."""
    return min(np.digitize(coord, bins) - 1, len(bins) - 2)

def find_throw_in_prob(x_coord, scoring_probabilities):
    """Find the throw-in goal probability from scoring probabilities."""
    for bin_range, prob_data in scoring_probabilities.items():
        # Remove 'm' from bin labels and split the range
        bin_start, bin_end = map(lambda x: int(x.replace('m', '')), bin_range.split('-'))
        if bin_start <= x_coord < bin_end:
            return prob_data["probability"] / 100  # Convert percentage to fraction
    return 0  # Default to 0 if no bin matches


def analyze_back_passes_vs_throw_ins(team, player_name, events_df, backward_pass_matrices, scoring_probabilities):
    """
    Compares back passes in the build-up third with throw-in probabilities from adjusted zones.
    Includes back passes where the probability is 0.
    """
    # Get the team's backward pass matrix
    backward_pass_matrix = backward_pass_matrices.get(team)
    if backward_pass_matrix is None:
        raise ValueError(f"Backward pass probability matrix for {team} not found.")
    
    # Filter back passes in the build-up third by the player
    back_passes = events_df[
        (events_df["team"] == team) &
        (events_df["player"] == player_name) &
        (events_df["type"] == "Pass") &
        (events_df["pass_angle"].apply(lambda angle: angle < -np.pi / 2 or angle > np.pi / 2)) &
        (events_df["location"].apply(lambda loc: isinstance(loc, list) and loc[0] <= 40))
    ]

    # Initialize counters and values
    total_back_passes = 0
    throw_in_greater_count = 0
    total_goal_prob_lost = 0

    for _, pass_event in back_passes.iterrows():
        # Find the bin for the pass end location
        end_location = pass_event["pass_end_location"]
        if not isinstance(end_location, list):  # Skip invalid locations
            continue
        
        x_bin = find_bin(end_location[0], x_bins)
        y_bin = find_bin(end_location[1], y_bins)

        # Get the back pass probability
        back_pass_prob = backward_pass_matrix[y_bin, x_bin]
        back_pass_prob = back_pass_prob / 100

        # Calculate the throw-in goal probability
        throw_in_x = end_location[0] + 60  # Shift x by 60
        throw_in_prob = find_throw_in_prob(throw_in_x, scoring_probabilities)

        # Update counts and values
        total_back_passes += 1
        if throw_in_prob > back_pass_prob:
            throw_in_greater_count += 1
            total_goal_prob_lost += throw_in_prob - back_pass_prob

    # Summary
    print(f"Analysis for {player_name} in {team}:")
    print(f"Total back passes in build-up third: {total_back_passes}")
    print(f"Back passes where throw-in value was greater: {throw_in_greater_count}")
    print(f"Total goal probability lost in these situations: {total_goal_prob_lost:.4f}")

    return {
        "total_back_passes": total_back_passes,
        "throw_in_greater_count": throw_in_greater_count,
        "total_goal_prob_lost": total_goal_prob_lost
    }

# Example usage
team = "Arsenal WFC"
player_name = "Katie McCabe"  # Replace with desired player's name

result = analyze_back_passes_vs_throw_ins(team, player_name, events_df, backward_pass_matrices, scoring_probabilities)
