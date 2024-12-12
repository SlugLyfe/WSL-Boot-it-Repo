from statsbombpy import sb
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings

# Suppress warnings
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

# Filter for possession data
possession_df = events_total_df.groupby(["match_id", "possession_team"]).size().reset_index(name="possession_count")

# Total possessions per match
match_totals = possession_df.groupby("match_id")["possession_count"].sum().reset_index()
match_totals.rename(columns={"possession_count": "total_possession"}, inplace=True)

# Merge with possession data
possession_df = possession_df.merge(match_totals, on="match_id")
possession_df["possession_percentage"] = (possession_df["possession_count"] / possession_df["total_possession"]) * 100

# Average possession across all matches for each team
average_possession = possession_df.groupby("possession_team")["possession_percentage"].mean().reset_index()
average_possession.rename(columns={"possession_team": "Team", "possession_percentage": "Average Possession (%)"}, inplace=True)

# Sort by possession percentage
average_possession_sorted = average_possession.sort_values(by="Average Possession (%)", ascending=False)

# Save results as a pickle file
with open("average_possession_results.pkl", "wb") as f:
    pickle.dump(average_possession_sorted, f)
print("Average possession results saved to average_possession_results.pkl.")

# Print the average possession for each team
print("\nAverage Possession by Team:")
for index, row in average_possession_sorted.iterrows():
    print(f"{row['Team']}: {row['Average Possession (%)']:.2f}%")

# Plot the average possession
fig, ax = plt.subplots(figsize=(12, 8))
x = range(len(average_possession_sorted))
ax.bar(x, average_possession_sorted["Average Possession (%)"], color="skyblue", width=0.6)
ax.set_xticks(x)
ax.set_xticklabels(average_possession_sorted["Team"], rotation=45, ha="right")
ax.set_ylabel("Average Possession (%)")
ax.set_title("Average Possession by Team")
plt.tight_layout()
plt.show()
