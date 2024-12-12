from statsbombpy import sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import warnings
import pickle 

# Suppress warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", message="credentials were not supplied. open data access only")

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
events_total_df = pd.concat(events_total_df, ignore_index=True)

# Define the list of defensive actions
defensive_actions = ["Pressure", "Tackle", "Interception", "Block", "Foul Committed"]

# Get unique teams
teams = events_total_df["team"].unique()

# Initialize dictionaries to store results
ppda_overall = {}
ppda_final_third = {}

# Debugging info
team_debug_info = {}

# Calculate PPDA
for team in teams:
    # Overall PPDA
    opponent_passes = events_total_df[(events_total_df["team"] != team) & (events_total_df["type"] == "Pass")]
    defensive_actions_team = events_total_df[(events_total_df["team"] == team) & (events_total_df["type"].isin(defensive_actions))]
    overall_defensive_actions_count = defensive_actions_team.shape[0]
    overall_opponent_passes_count = opponent_passes.shape[0]
    overall_ppda = overall_opponent_passes_count / overall_defensive_actions_count if overall_defensive_actions_count > 0 else float('inf')
    ppda_overall[team] = overall_ppda

    # Final Third PPDA
    final_third_opponent_passes = opponent_passes[opponent_passes["location"].apply(lambda loc: loc[0] > 80 if isinstance(loc, list) else False)]
    final_third_defensive_actions = defensive_actions_team[defensive_actions_team["location"].apply(lambda loc: loc[0] > 80 if isinstance(loc, list) else False)]
    final_third_defensive_actions_count = final_third_defensive_actions.shape[0]
    final_third_opponent_passes_count = final_third_opponent_passes.shape[0]
    final_ppda = final_third_opponent_passes_count / final_third_defensive_actions_count if final_third_defensive_actions_count > 0 else float('inf')
    ppda_final_third[team] = final_ppda

    # Debugging info
    team_debug_info[team] = {
        "Overall Defensive Actions": overall_defensive_actions_count,
        "Overall Opponent Passes": overall_opponent_passes_count,
        "Final Third Defensive Actions": final_third_defensive_actions_count,
        "Final Third Opponent Passes": final_third_opponent_passes_count,
    }

# Print debugging info
print("\nDebugging Info:")
for team, counts in team_debug_info.items():
    print(f"Team: {team}")
    print(f"  Overall Defensive Actions: {counts['Overall Defensive Actions']}")
    print(f"  Overall Opponent Passes: {counts['Overall Opponent Passes']}")
    print(f"  Final Third Defensive Actions: {counts['Final Third Defensive Actions']}")
    print(f"  Final Third Opponent Passes: {counts['Final Third Opponent Passes']}")

# Sort and print results
sorted_ppda_overall = sorted(ppda_overall.items(), key=lambda x: x[1])
sorted_ppda_final_third = sorted(ppda_final_third.items(), key=lambda x: x[1])

print("\nPPDA Results Overall (lower is better):")
for team, ppda in sorted_ppda_overall:
    print(f"{team}: {ppda:.2f}")

print("\nPPDA Results in Final Third (lower is better):")
for team, ppda in sorted_ppda_final_third:
    print(f"{team}: {ppda:.2f}")

# Save results as pickle files
with open("ppda_overall_results.pkl", "wb") as f:
    pickle.dump(ppda_overall, f)

with open("ppda_final_third_results.pkl", "wb") as f:
    pickle.dump(ppda_final_third, f)

print("\nPPDA results saved to ppda_overall_results.pkl and ppda_final_third_results.pkl.")

# Create bar chart for Final Third PPDA visualization
final_third_ppda_df = pd.DataFrame(sorted_ppda_final_third, columns=["Team", "PPDA Final Third"])

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(final_third_ppda_df))
ax.bar(x, final_third_ppda_df["PPDA Final Third"], width=0.6, color="orange", label="Final Third PPDA")
ax.set_xticks(x)
ax.set_xticklabels(final_third_ppda_df["Team"], rotation=45, ha="right")
ax.set_ylabel("PPDA in Final Third")
ax.set_title("PPDA in Final Third by Team")
ax.legend()
plt.tight_layout()
plt.show()

