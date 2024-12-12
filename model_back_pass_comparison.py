import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load necessary data
with open("backward_pass_probability_matrices.pkl", "rb") as f:
    backward_pass_matrices = pickle.load(f)

with open("defending_throw_in_probability_matrices.pkl", "rb") as f:
    throw_in_matrices = pickle.load(f)

with open("events_with_xt.pkl", "rb") as f:
    events_df = pickle.load(f)

with open("scoring_probabilities.pkl", "rb") as f:
    scoring_probabilities = pickle.load(f)

with open("trained_linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load additional data
with open("average_possession_results.pkl", "rb") as f:
    average_possession = pickle.load(f).set_index("Team")["Average Possession (%)"].to_dict()

with open("average_pressure_faced_final_third.pkl", "rb") as f:
    apf_final_third = pickle.load(f)

with open("wsl_team_stats.pkl", "rb") as f:
    team_stats = pickle.load(f)

xT_map = np.load("xT_map.npy")

# Define pitch dimensions and bins
x_bins = np.linspace(0, 105, 17)  # 16 bins for x
y_bins = np.linspace(0, 68, 13)  # 12 bins for y

def find_bin(coord, bins):
    """Helper function to find the bin index for a coordinate."""
    return min(np.digitize(coord, bins) - 1, len(bins) - 2)

def find_throw_in_prob(x_coord, scoring_probabilities):
    """Find the throw-in goal probability from scoring probabilities."""
    for bin_range, prob_data in scoring_probabilities.items():
        bin_start, bin_end = map(lambda x: int(x.replace('m', '')), bin_range.split('-'))
        if bin_start <= x_coord < bin_end:
            return prob_data["probability"] / 100  # Convert percentage to fraction
    return 0  

def analyze_model_decisions_vs_player(team, player_name, events_df, backward_pass_matrices, scoring_probabilities, model):
    """
    Compares the model's decisions to the player's back pass behavior in the build-up third.
    Uses the model's predictions to decide between back pass or throw-in.
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
    total_goal_prob_diff = 0  

    for _, pass_event in back_passes.iterrows():
        # Find the bin for the pass end location
        end_location = pass_event["pass_end_location"]
        if not isinstance(end_location, list):  # Skip invalid locations
            continue

        x_bin = find_bin(end_location[0], x_bins)
        y_bin = find_bin(end_location[1], y_bins)

        # Get the back pass probability from the team-specific matrix
        back_pass_prob = backward_pass_matrix[y_bin, x_bin]

        # Calculate the throw-in goal probability
        throw_in_x = end_location[0] + 60  # Shift x by 60
        throw_in_prob = find_throw_in_prob(throw_in_x, scoring_probabilities)

        # Dynamically calculate xT value from the map
        try:
            xT_value = xT_map[y_bin, x_bin]
        except IndexError:
            print(f"Invalid xT map indices for pass at x_bin={x_bin}, y_bin={y_bin}")
            continue

        # Get other feature values for the model
        avg_possession = average_possession[team]
        apf_value = apf_final_third[team]
        avg_position = team_stats[team]["Average Position"]
        avg_goals_per_game = team_stats[team]["Average Goals Per Game"]

        # Create the feature vector for the model
        model_features = np.array([[xT_value, avg_possession, apf_value, avg_position, avg_goals_per_game]])
        model_pred = model.predict(model_features)[0]  # Model's predicted probability
        model_pred = model_pred / 100

        # Debugging throw-in probability and model predictions
        print(f"Pass location x: {end_location[0]}, Throw-in x: {throw_in_x}, Throw-in prob: {throw_in_prob}")
        print(f"Model prediction: {model_pred}")

        # Update counts and values based on model's decision
        total_back_passes += 1
        total_goal_prob_diff += abs(throw_in_prob - model_pred)

        # Compare throw-in probability to the team-specific back-pass probability
        if throw_in_prob > back_pass_prob:
            throw_in_greater_count += 1
            total_goal_prob_lost += throw_in_prob - back_pass_prob

    # Summary
    print(f"Model replacing {player_name} in {team}:")
    print(f"Total back passes in build-up third: {total_back_passes}")
    print(f"Model chose throw-in over back pass: {throw_in_greater_count}")
    print(f"Total goal probability lost in these situations: {total_goal_prob_lost:.4f}")
   # print(f"Total goal probability difference (all comparisons): {total_goal_prob_diff:.4f}")

    return {
        "total_back_passes": total_back_passes,
        "throw_in_greater_count": throw_in_greater_count,
        "total_goal_prob_lost": total_goal_prob_lost,
        "total_goal_prob_diff": total_goal_prob_diff
    }

# Example usage
team = "Arsenal WFC"
player_name = "Katie McCabe"  

result = analyze_model_decisions_vs_player(
    team, player_name, events_df, backward_pass_matrices, scoring_probabilities, model
)
