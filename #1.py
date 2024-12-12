# Importing necessary libraries
from statsbombpy import sb
import pandas as pd
import numpy as np
from mplsoccer import Pitch
import matplotlib.pyplot as plt

# Fetch WSL matches across three seasons
wsl_1819_matches = sb.matches(competition_id=37, season_id=4)
wsl_1920_matches = sb.matches(competition_id=37, season_id=42)
wsl_2021_matches = sb.matches(competition_id=37, season_id=90)
wsl_total_matches = pd.concat([wsl_1819_matches, wsl_1920_matches, wsl_2021_matches], ignore_index=True)
wsl_total_matches_sorted_by_date = wsl_total_matches.sort_values(by="match_date", ascending=True)

# Fetch all events for the matches and create a combined events dataframe
events_total_df = []
for this_match_id in wsl_total_matches_sorted_by_date["match_id"]:
    events_for_a_game_df = sb.events(match_id=this_match_id)
    events_for_a_game_df["match_id"] = this_match_id
    events_total_df.append(events_for_a_game_df)

events_total_df = pd.concat(events_total_df, ignore_index=True)

# Load the precomputed xT map from file
xT_map = np.load("xT_map.npy")

# Function to calculate xT for an event
def calculate_xt_from_event(event, xT_map, pitch_length=105, pitch_width=68):
    """
    Calculate the Expected Threat (xT) value from an event's location using a precomputed xT map.
    """
    location = event.get("location") if isinstance(event, dict) else event["location"]
    
    # Ensure location is valid
    if isinstance(location, list) and len(location) == 2:  # Valid location with x, y coordinates
        x, y = location[0] * pitch_length / 120, location[1] * pitch_width / 80  # Convert StatsBomb coords to pitch coords
        bin_x = min(int(x // (pitch_length / xT_map.shape[1])), xT_map.shape[1] - 1)
        bin_y = min(int(y // (pitch_width / xT_map.shape[0])), xT_map.shape[0] - 1)
        return xT_map[bin_y, bin_x]
    
    # Return 0 if location is invalid or missing
    return 0.0

# Add xT values to events dataframe
def add_xt_to_events(events_df, xT_map):
    """
    Add a column to the events dataframe with the xT value for each event.
    """
    events_df["xT"] = events_df.apply(lambda row: calculate_xt_from_event(row, xT_map), axis=1)
    return events_df

# Add xT to all events
events_with_xt = add_xt_to_events(events_total_df, xT_map)

#test
passes = events_with_xt[events_with_xt["type"] == "Pass"]
print(passes[["player", "location", "xT"]])

# Save the processed dataframe as a pickle file
events_with_xt.to_pickle("events_with_xt.pkl")
print("events_with_xt saved to 'events_with_xt.pkl'.")
