import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

teams = pd.read_csv("teams.csv")

teams

teams = teams [["team", "country", "year", "events", "athletes", "age","medals", "prev_medals"]]

# Only keep numeric columns
numeric_teams = teams.select_dtypes(include=[np.number])

# Now calculate correlation
correlation_with_medals = numeric_teams.corr()['medals']

print(correlation_with_medals)

# Scatter plot
plt.scatter(teams['athletes'], teams['medals'], color='blue')

# Fit a regression line
coeffs = np.polyfit(teams['athletes'], teams['medals'], 1)
regression_line = np.poly1d(coeffs)

# Plot regression line
plt.plot(teams['athletes'], regression_line(teams['athletes']), color='red')

# Labels and title
plt.xlabel('Number of Athletes')
plt.ylabel('Number of Medals')
plt.title('Relationship between Athletes and Medals')
plt.grid(True)
plt.show()

# Scatter plot
plt.scatter(teams['age'], teams['medals'], color='blue')

# Fit a regression line
coeffs = np.polyfit(teams['age'], teams['medals'], 1)
regression_line = np.poly1d(coeffs)

# Plot regression line
plt.plot(teams['age'], regression_line(teams['age']), color='red')

# Labels and title
plt.xlabel('Age')
plt.ylabel('Number of Medals')
plt.title('Relationship between Age and Medals')
plt.grid(True)
plt.show()

teams.plot.hist(y="medals")

teams[teams.isnull().any(axis=1)]

train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

predictors = ["athletes", "prev_medals"]
target = "medals"

# Always clean first
train = train.dropna(subset=predictors + [target])
test = test.dropna(subset=predictors + [target])

# Optional: reset index (to avoid index mismatches)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Now prepare X_train, y_train
X_train = np.column_stack((np.ones(train.shape[0]), train[predictors].values))
y_train = train[target].values

# Prepare training data
X_train = np.column_stack((np.ones(train.shape[0]), train[predictors].values))
y_train = train[target].values

# Calculate coefficients (beta)
beta, residuals, rank, s = np.linalg.lstsq(X_train, y_train, rcond=None)

beta, residuals, rank, s = np.linalg.lstsq(X_train, y_train, rcond=None)

# Prepare testing data
X_test = np.column_stack((np.ones(test.shape[0]), test[predictors].values))

# Predict
predictions = X_test @ beta

# Save predictions in test DataFrame
test["predictions"] = predictions

# Calculate RMSE
rmse = np.sqrt(np.mean((test[target] - test["predictions"]) ** 2))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

import matplotlib.pyplot as plt

plt.scatter(test[target], test["predictions"], color='blue')
plt.plot([test[target].min(), test[target].max()], [test[target].min(), test[target].max()], color='red', linewidth=2)
plt.xlabel('Actual Medals')
plt.ylabel('Predicted Medals')
plt.title('Actual vs Predicted Medals')
plt.grid(True)
plt.show()

# R-squared calculation
SS_res = np.sum((test[target] - test["predictions"]) ** 2)
SS_tot = np.sum((test[target] - np.mean(test[target])) ** 2)
r_squared = 1 - (SS_res / SS_tot)

print(f"R-squared: {r_squared:.2f}")

# Add error column
test["error"] = test["predictions"] - test["medals"]

# View sorted by error
test_sorted = test.sort_values("error", ascending=False)

# Display
print(test_sorted[["team", "year", "athletes", "prev_medals", "medals", "predictions", "error"]])

