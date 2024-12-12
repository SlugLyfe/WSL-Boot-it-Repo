import pandas as pd
import numpy as np
import pickle

# Load the processed dataframe
events_with_xt = pd.read_pickle("events_with_xt.pkl")

# Display basic dataset information
print(f"Total events in the dataset: {len(events_with_xt)}")
print("Columns in the dataset:")
print(events_with_xt.columns)

# Define bins and labels for throw-in distance
bins = [60, 75, 90, 105, 120]
bin_labels = ["60-75m", "75-90m", "90-105m", "105-120m"]

# Filter for throw-ins using play_pattern field
throw_ins = events_with_xt[
    (events_with_xt["type"] == "Pass") & 
    (events_with_xt["play_pattern"] == "From Throw In") &
    (events_with_xt["location"].apply(lambda loc: isinstance(loc, list) and loc[0] >= 60))
]

print(f"Total throw-ins in opponent's half: {len(throw_ins)}")
print(throw_ins[["type", "location", "play_pattern", "team"]].head(20))

# Efficiently bin throw-ins by distance
throw_ins = throw_ins.copy()  # Avoid SettingWithCopyWarning
throw_ins["distance_bin"] = pd.cut(
    throw_ins["location"].apply(lambda loc: loc[0]), bins=bins, labels=bin_labels, right=False
)
throw_in_distribution = throw_ins["distance_bin"].value_counts().sort_index()
print("\nThrow-in distribution by bins:")
print(throw_in_distribution)

# Function to determine if a throw-in is pressured
def is_throw_in_pressured(events_df, throw_in_index):
    """
    Determines if a throw-in is pressured based on the following conditions:
    1. The throw-in itself is under pressure.
    2. The first pass after the throw-in is under pressure.
    3. The second pass after the throw-in is under pressure (if the first pass is not a long ball).
    """
    throw_in = events_df.iloc[throw_in_index]

    # Condition 1: Throw-in itself is under pressure
    if throw_in.get("under_pressure", False):
        return True

    # Get the next two actions
    next_actions = events_df.iloc[throw_in_index + 1 : throw_in_index + 3]
    
    if next_actions.empty:
        return False

    # Condition 2: First pass after the throw-in is under pressure
    first_action = next_actions.iloc[0]
    if first_action.get("under_pressure", False):
        return True

    # Condition 3: Second pass under pressure if the first pass is not a long ball
    if len(next_actions) > 1:
        second_action = next_actions.iloc[1]
        if (
            first_action["type"] == "Pass"
            and first_action.get("subtype") != "Long Ball"
            and second_action.get("under_pressure", False)
        ):
            return True

    return False

# Filter for pressured throw-ins based on the new criteria
pressured_throw_ins = throw_ins[
    throw_ins.index.map(lambda idx: is_throw_in_pressured(events_with_xt, idx))
]

print(f"\nTotal pressured throw-ins: {len(pressured_throw_ins)}")

# Link pressured throw-ins to goals scored by the defending team within 10 moves
scoring_probabilities = {}

for bin_label in bin_labels:
    bin_throw_ins = pressured_throw_ins[pressured_throw_ins["distance_bin"] == bin_label]
    total_throw_ins = len(bin_throw_ins)

    # Aggregate all possession chains for the bin
    goals_scored = 0
    if total_throw_ins > 0:
        for _, throw_in in bin_throw_ins.iterrows():
            # Filter possession chain events for this throw-in
            possession_chain = events_with_xt[
                (events_with_xt["possession"] == throw_in["possession"]) &
                (events_with_xt["index"] >= throw_in["index"]) &
                (events_with_xt["index"] <= throw_in["index"] + 10)
            ]

            # Identify the defending team
            defending_team = events_with_xt.loc[
                (events_with_xt["match_id"] == throw_in["match_id"]) & 
                (events_with_xt["team"] != throw_in["team"]), 
                "team"
            ].iloc[0]

            # Check if the defending team scored a goal
            if (
                not possession_chain.empty and
                (possession_chain["type"] == "Shot").any() and
                (possession_chain[(possession_chain["type"] == "Shot") & 
                                  (possession_chain["shot_outcome"] == "Goal")]["team"] == defending_team).any()
            ):
                goals_scored += 1

    scoring_probabilities[bin_label] = {
        "goals_scored": goals_scored,
        "total_throw_ins": total_throw_ins,
        "probability": (goals_scored / total_throw_ins) * 100 if total_throw_ins > 0 else 0
    }

# Print results
print("\nScoring probabilities after pressured throw-ins (by distance bin, defending team):")
for bin_label, stats in scoring_probabilities.items():
    print(f"{bin_label}: {stats['probability']:.2f}% ({stats['goals_scored']} goals out of {stats['total_throw_ins']} throw-ins)")

# Save the dictionary to a pickle file
with open("scoring_probabilities.pkl", "wb") as pickle_file:
    pickle.dump(scoring_probabilities, pickle_file)
print("Scoring probabilities saved to scoring_probabilities.pkl.")