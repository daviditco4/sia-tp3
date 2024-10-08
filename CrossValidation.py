import pandas as pd
import numpy as np
from LinearPerceptron import LinearPerceptron
from SigmoidPerceptron import SigmoidPerceptron
from sklearn.model_selection import KFold
from Perceptron import min_max_normalize_output, min_max_normalize

# Accuracy calculation function
def accuracy(predictions, actual):
    correct = np.sum(predictions == actual)
    total = len(actual)
    return correct / total

# Read the CSV file and prepare the data
df = pd.read_csv('exercise2\data\set.csv')
X = df.to_numpy()
y = X[:, -1]  # Last column as target labels
X = X[:, :-1]  # All columns except last as features

norm_X, min_x, max_x = min_max_normalize(X)
norm_Y, min_y, max_y = min_max_normalize_output(y)

# Add a column of ones to account for bias
ones_column = np.ones((norm_X.shape[0], 1))
norm_X = np.hstack((ones_column, norm_X))

# Initialize cross-validation with 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store accuracies for each fold
linear_train_accuracies = []
linear_test_accuracies = []
nonlinear_train_accuracies = []
nonlinear_test_accuracies = []
errors_linear = []
errors_nonlinear = []

# Perform cross-validation
for train_index, test_index in kf.split(norm_X):
    # Split data into training and test sets for this fold
    X_train, X_test = norm_X[train_index], norm_X[test_index]
    y_train, y_test = norm_Y[train_index], norm_Y[test_index]

    # Train the Linear Perceptron on training data
    linear_perceptron = LinearPerceptron(learning_rate=0.1, max_iterations=1000)
    linear_perceptron.fit(X_train, y_train)
    
    # Train the Nonlinear Perceptron on training data
    nonlinear_perceptron = SigmoidPerceptron(learning_rate=0.1, max_iterations=1000)
    nonlinear_perceptron.fit(X_train, y_train)

    # Evaluate Linear Perceptron on both training and test data
    linear_train_predictions = linear_perceptron.predict(X_train)
    linear_test_predictions = linear_perceptron.predict(X_test)
    linear_train_accuracies.append(accuracy(linear_train_predictions, y_train))
    linear_test_accuracies.append(accuracy(linear_test_predictions, y_test))

    # Evaluate Nonlinear Perceptron on both training and test data
    nonlinear_train_predictions = nonlinear_perceptron.predict(X_train)
    nonlinear_test_predictions = nonlinear_perceptron.predict(X_test)
    nonlinear_train_accuracies.append(accuracy(nonlinear_train_predictions, y_train))
    nonlinear_test_accuracies.append(accuracy(nonlinear_test_predictions, y_test))

# Calculate average accuracy across all folds
avg_linear_train_accuracy = np.mean(linear_train_accuracies)
avg_linear_test_accuracy = np.mean(linear_test_accuracies)
avg_nonlinear_train_accuracy = np.mean(nonlinear_train_accuracies)
avg_nonlinear_test_accuracy = np.mean(nonlinear_test_accuracies)

print(f"Linear Perceptron Average Training Accuracy: {avg_linear_train_accuracy}")
print(f"Linear Perceptron Average Test Accuracy: {avg_linear_test_accuracy}")
print(f"Nonlinear Perceptron Average Training Accuracy: {avg_nonlinear_train_accuracy}")
print(f"Nonlinear Perceptron Average Test Accuracy: {avg_nonlinear_test_accuracy}")