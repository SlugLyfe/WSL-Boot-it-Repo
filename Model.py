import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the xT map
xT_map = np.load("xT_map.npy")  # General xT value map

# Load the average possession statistics
with open("average_possession_results.pkl", "rb") as f:
    average_possession = pickle.load(f).set_index("Team")["Average Possession (%)"].to_dict()

# Load the APF (Final Third) statistics
with open("average_pressure_faced_final_third.pkl", "rb") as f:
    apf_final_third = pickle.load(f)  # Dictionary: {team: APF value}

# Load the backward pass probability matrices
with open("backward_pass_probability_matrices.pkl", "rb") as f:
    backward_pass_matrices = pickle.load(f)  # Dictionary: {team: matrix}

# Load the team statistics (average league position and goals per game)
with open("wsl_team_stats.pkl", "rb") as f:
    team_stats = pickle.load(f)  # Dictionary: {team: {"Average Position": ..., "Average Goals Per Game": ...}}

# Prepare the data for linear regression
features = []
targets = []

teams = backward_pass_matrices.keys()
x_bins = np.linspace(0, 105, 17)  # 16 bins for x
y_bins = np.linspace(0, 68, 13)  # 12 bins for y

for team in teams:
    if team in average_possession and team in apf_final_third and team in team_stats:
        backward_pass_matrix = backward_pass_matrices[team]

        for i in range(backward_pass_matrix.shape[0]):
            for j in range(backward_pass_matrix.shape[1]):
                goal_prob = backward_pass_matrix[i, j]  # Target value
                if 1 <= goal_prob <= 9:  # Filter for values between 1% and 9%
                    xT_value = xT_map[i, j]  # xT for the zone
                    avg_position = team_stats[team]["Average Position"]  # Average league position
                    avg_goals_per_game = team_stats[team]["Average Goals Per Game"]  # Goals per game

                    # Append all features
                    features.append([
                        xT_value,
                        average_possession[team],
                        apf_final_third[team],
                        avg_position,
                        avg_goals_per_game
                    ])
                    targets.append(goal_prob)

# Check if sufficient data points exist
if len(features) < 2:
    print("Insufficient data points for meaningful analysis. Check the input data or filtering conditions.")
else:
    # Convert features and targets to numpy arrays
    X = np.array(features)
    y = np.array(targets)

    # Split the data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=865)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Linear Regression Model Evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")

    # Display the coefficients
    feature_names = ["xT", "Average Possession (%)", "APF", "Average League Position", "Average Goals Per Game"]
    coefficients = model.coef_
    intercept = model.intercept_

    print("\nModel Coefficients:")
    for name, coef in zip(feature_names, coefficients):
        print(f"{name}: {coef:.4f}")
    print(f"Intercept: {intercept:.4f}")

    # Define colors based on accuracy
    absolute_error = np.abs(y_test - y_pred)
    relative_error = absolute_error / y_test
    colors = []
    for error in relative_error:
        if error > 0.2:  # Red for predictions >20% off
            colors.append("red")
        else:
            green_intensity = max(0, min(int((1 - error / 0.2) * 255), 255))  # Clamp green intensity
            colors.append(f"#{green_intensity:02x}ff{green_intensity:02x}")

    # Scatter plot with accuracy-based colors and 50% transparency
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, c=colors, edgecolor="black", s=70, label="Predicted vs Actual", alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2, label="Perfect Prediction Line")
    plt.xlabel("Goal Probability from Backward Pass (%)")
    plt.ylabel("Predicted Goal Probability (%)")
    plt.title("Linear Regression: Actual vs Predicted Goal Probability")
    plt.legend()

    # Add accuracy scale
    cmap = ListedColormap(["green", "red"])
    norm = plt.Normalize(vmin=0, vmax=0.4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation="vertical", fraction=0.05, pad=0.05)
    cbar.set_label("Prediction Accuracy (Green: Accurate, Red: Inaccurate)", fontsize=10)

    plt.tight_layout()
    plt.show()

    # Save the trained linear regression model to a file
with open("trained_linear_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Trained linear regression model saved as 'trained_linear_regression_model.pkl'.")

