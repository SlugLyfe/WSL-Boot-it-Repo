import pandas as pd
import numpy as np
import pickle

# Load the processed dataframe
events_with_xt = pd.read_pickle("events_with_xt.pkl")

# Load the PPDA values for each team in the final third
with open("ppda_final_third_results.pkl", "rb") as f:
    ppda_final_third_dict = pickle.load(f)  # Ensure this dictionary contains final third PPDA values for all teams

# Define the bins for the build-up third
build_up_threshold = 40  # x-coordinate threshold for the build-up third

def calculate_apf_final_third(events_df, ppda_final_third_dict):
    """
    Calculates Average Pressure Faced (APF) using final third PPDA for each team.
    """
    teams = events_df["team"].unique()
    apf_results = {}

    for team in teams:
        # Filter backward passes for the team in the build-up third
        backward_passes = events_df[
            (events_df["team"] == team) &
            (events_df["type"] == "Pass") &
            (events_df["pass_angle"].apply(lambda angle: angle < -np.pi / 2 or angle > np.pi / 2)) &
            (events_df["location"].apply(lambda loc: isinstance(loc, list) and loc[0] <= build_up_threshold))
        ]

        if backward_passes.empty:
            apf_results[team] = 0  # No backward passes, APF = 0
            continue

        # Calculate the fraction of backward passes against each opponent
        opponent_fractions = {}
        for _, pass_event in backward_passes.iterrows():
            # Find the opponent for this pass
            match_id = pass_event["match_id"]
            opponent = events_df[
                (events_df["match_id"] == match_id) & 
                (events_df["team"] != team)
            ]["team"].iloc[0]

            if opponent not in opponent_fractions:
                opponent_fractions[opponent] = 0
            opponent_fractions[opponent] += 1

        # Normalize the fractions
        total_backward_passes = len(backward_passes)
        for opponent in opponent_fractions:
            opponent_fractions[opponent] /= total_backward_passes

        # Calculate APF for the team using final third PPDA
        apf = 0
        for opponent, fraction in opponent_fractions.items():
            ppda_final_third = ppda_final_third_dict.get(opponent, 0)  # Get final third PPDA for the opponent
            apf += ppda_final_third * fraction

        apf_results[team] = apf

    return apf_results

# Calculate APF using final third PPDA for all teams
apf_final_third_results = calculate_apf_final_third(events_with_xt, ppda_final_third_dict)

# Print the results
print("\nAverage Pressure Faced (APF - Final Third PPDA) for Each Team:")
for team, apf in apf_final_third_results.items():
    print(f"{team}: {apf:.2f}")

# Save the APF results to a pickle file
with open("average_pressure_faced_final_third.pkl", "wb") as f:
    pickle.dump(apf_final_third_results, f)

print("\nAPF (Final Third PPDA) results saved to average_pressure_faced_final_third.pkl.")
