import pandas as pd
import numpy as np
from LinearPerceptron import LinearPerceptron
from NonLinearPerceptron import NonlinearPerceptron
from sklearn.model_selection import KFold

# Mean Squared Error calculation function
def mean_squared_error(predictions, actual):
    return np.mean((predictions - actual) ** 2)

# Read the CSV file and prepare the data
df = pd.read_csv('TP3-ej2-conjunto.csv')
X = df.to_numpy()
y = X[:, -1]  # Last column as target values
X = X[:, :-1]  # All columns except last as features

# Normalize y to range [0, 1]
y_min, y_max = y.min(), y.max()
y = (y - y_min) / (y_max - y_min)

# Add a column of ones to account for bias
ones_column = np.ones((X.shape[0], 1))
X = np.hstack((ones_column, X))

# Normalize features to range [-1, 1]
epsilon = 1e-8
X = 2 * (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + epsilon) - 1

# Initialize cross-validation with 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store MSE for each fold
linear_train_mse = []
linear_test_mse = []
nonlinear_train_mse = []
nonlinear_test_mse = []

# Perform cross-validation
for train_index, test_index in kf.split(X):
    # Split data into training and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the Linear Perceptron on training data
    linear_perceptron = LinearPerceptron(learning_rate=0.025, max_iterations=10000)
    linear_perceptron.fit(X_train, y_train)
    
    # Train the Nonlinear Perceptron on training data
    nonlinear_perceptron = NonlinearPerceptron(learning_rate=0.025, max_iterations=10000)
    nonlinear_perceptron.fit(X_train, y_train)

    # Evaluate Linear Perceptron on both training and test data
    linear_train_predictions = linear_perceptron.predict(X_train)
    linear_test_predictions = linear_perceptron.predict(X_test)
    linear_train_mse.append(mean_squared_error(linear_train_predictions, y_train))
    linear_test_mse.append(mean_squared_error(linear_test_predictions, y_test))

    # Evaluate Nonlinear Perceptron on both training and test data
    nonlinear_train_predictions = nonlinear_perceptron.predict(X_train)
    nonlinear_test_predictions = nonlinear_perceptron.predict(X_test)
    nonlinear_train_mse.append(mean_squared_error(nonlinear_train_predictions, y_train))
    nonlinear_test_mse.append(mean_squared_error(nonlinear_test_predictions, y_test))

# Calculate average MSE across all folds
avg_linear_train_mse = np.mean(linear_train_mse)
avg_linear_test_mse = np.mean(linear_test_mse)
avg_nonlinear_train_mse = np.mean(nonlinear_train_mse)
avg_nonlinear_test_mse = np.mean(nonlinear_test_mse)

print(f"Linear Perceptron Average Training MSE: {avg_linear_train_mse}")
print(f"Linear Perceptron Average Test MSE: {avg_linear_test_mse}")
print(f"Nonlinear Perceptron Average Training MSE: {avg_nonlinear_train_mse}")
print(f"Nonlinear Perceptron Average Test MSE: {avg_nonlinear_test_mse}")