import numpy as np
import pickle

# Mapping team names to match StatsBomb data
team_name_map = {
    "Arsenal": "Arsenal WFC",
    "Man City": "Manchester City WFC",
    "Chelsea": "Chelsea FCW",
    "Birmingham City": "Birmingham City WFC",
    "Reading": "Reading FC Women",
    "Bristol City": "Bristol City WFC",
    "West Ham": "West Ham United WFC",
    "Liverpool": "Liverpool FC Women",
    "Brighton & Hove Albion": "Brighton & Hove Albion WFC",
    "Everton": "Everton WFC",
    "Yeovil Town": "Yeovil Town LFC",
    "Man Utd": "Manchester United WFC",
    "Tottenham": "Tottenham Hotspur Women",
    "Aston Villa": "Aston Villa WFC",
}

# Data for 2018-19 season
season_2018_19 = {
    "Arsenal": {"position": 1, "GF": 70, "MP": 20},
    "Man City": {"position": 2, "GF": 53, "MP": 20},
    "Chelsea": {"position": 3, "GF": 46, "MP": 20},
    "Birmingham City": {"position": 4, "GF": 29, "MP": 20},
    "Reading": {"position": 5, "GF": 33, "MP": 20},
    "Bristol City": {"position": 6, "GF": 17, "MP": 20},
    "West Ham": {"position": 7, "GF": 25, "MP": 20},
    "Liverpool": {"position": 8, "GF": 21, "MP": 20},
    "Brighton & Hove Albion": {"position": 9, "GF": 16, "MP": 20},
    "Everton": {"position": 10, "GF": 13, "MP": 20},
    "Yeovil Town": {"position": 11, "GF": 10, "MP": 20},
}

# Data for 2019-20 season
season_2019_20 = {
    "Chelsea": {"position": 1, "GF": 47, "MP": 15},
    "Man City": {"position": 2, "GF": 39, "MP": 16},
    "Arsenal": {"position": 3, "GF": 40, "MP": 15},
    "Man Utd": {"position": 4, "GF": 24, "MP": 14},
    "Reading": {"position": 5, "GF": 21, "MP": 14},
    "Everton": {"position": 6, "GF": 21, "MP": 14},
    "Tottenham": {"position": 7, "GF": 15, "MP": 15},
    "West Ham": {"position": 8, "GF": 19, "MP": 14},
    "Brighton & Hove Albion": {"position": 9, "GF": 11, "MP": 16},
    "Bristol City": {"position": 10, "GF": 9, "MP": 14},
    "Birmingham City": {"position": 11, "GF": 5, "MP": 13},
    "Liverpool": {"position": 12, "GF": 8, "MP": 14},
}

# Data for 2020-21 season
season_2020_21 = {
    "Chelsea": {"position": 1, "GF": 69, "MP": 22},
    "Man City": {"position": 2, "GF": 65, "MP": 22},
    "Arsenal": {"position": 3, "GF": 63, "MP": 22},
    "Man Utd": {"position": 4, "GF": 44, "MP": 22},
    "Everton": {"position": 5, "GF": 39, "MP": 22},
    "Brighton & Hove Albion": {"position": 6, "GF": 21, "MP": 22},
    "Reading": {"position": 7, "GF": 25, "MP": 22},
    "Tottenham": {"position": 8, "GF": 18, "MP": 22},
    "West Ham": {"position": 9, "GF": 21, "MP": 22},
    "Aston Villa": {"position": 10, "GF": 15, "MP": 22},
    "Birmingham City": {"position": 11, "GF": 15, "MP": 22},
    "Bristol City": {"position": 12, "GF": 18, "MP": 22},
}

# Combine all seasons
seasons = [season_2018_19, season_2019_20, season_2020_21]

# Initialize combined dictionary
teams_stats = {}

for season in seasons:
    for team, stats in season.items():
        standardized_team = team_name_map.get(team, team)  # Standardize team names
        if standardized_team not in teams_stats:
            teams_stats[standardized_team] = {"total_position": 0, "total_GF": 0, "total_MP": 0, "seasons": 0}
        teams_stats[standardized_team]["total_position"] += stats["position"]
        teams_stats[standardized_team]["total_GF"] += stats["GF"]
        teams_stats[standardized_team]["total_MP"] += stats["MP"]
        teams_stats[standardized_team]["seasons"] += 1

# Calculate average position and goals per game
teams_summary = {}
for team, stats in teams_stats.items():
    avg_position = stats["total_position"] / stats["seasons"]
    avg_goals_per_game = stats["total_GF"] / stats["total_MP"]
    teams_summary[team] = {
        "Average Position": round(avg_position, 2),
        "Average Goals Per Game": round(avg_goals_per_game, 2),
    }

# Save results as a pickle file
with open("wsl_team_stats.pkl", "wb") as f:
    pickle.dump(teams_summary, f)

print("\nWSL team stats saved to wsl_team_stats.pkl")

# Print the average position and goals per game
for team, data in teams_summary.items():
    print(f"{team}: Avg Position: {data['Average Position']:.2f}, Avg Goals Per Game: {data['Average Goals Per Game']:.2f}")
