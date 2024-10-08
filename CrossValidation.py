import pandas as pd
import numpy as np
from LinearPerceptron import LinearPerceptron
from NonLinearPerceptron import NonlinearPerceptron
from sklearn.model_selection import KFold

# Accuracy calculation function
def accuracy(predictions, actual):
    correct = np.sum(predictions == actual)
    total = len(actual)
    return correct / total

# Read the CSV file and prepare the data
df = pd.read_csv('TP3-ej2-conjunto.csv')
X = df.to_numpy()
y = X[:, -1]  # Last column as target labels
X = X[:, :-1]  # All columns except last as features

# Add a column of ones to account for bias
ones_column = np.ones((X.shape[0], 1))
X = np.hstack((ones_column, X))

# Normalize features to range [-1, 1]
epsilon = 1e-8
X = 2 * (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + epsilon) - 1

# Convert continuous labels to binary (-1 and 1)
median_value = np.median(y)
y_binary = np.where(y > median_value, 1, -1)

# Initialize cross-validation with 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store accuracies for each fold
linear_train_accuracies = []
linear_test_accuracies = []
nonlinear_train_accuracies = []
nonlinear_test_accuracies = []

# Perform cross-validation
for train_index, test_index in kf.split(X):
    # Split data into training and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_binary[train_index], y_binary[test_index]

    # Train the Linear Perceptron on training data
    linear_perceptron = LinearPerceptron(learning_rate=0.025, max_iterations=10000)
    linear_perceptron.fit(X_train, y_train)
    
    # Train the Nonlinear Perceptron on training data
    nonlinear_perceptron = NonlinearPerceptron(learning_rate=0.025, max_iterations=10000)
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