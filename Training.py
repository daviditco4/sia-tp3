import pandas as pd
import numpy as np
from LinearPerceptron import LinearPerceptron
from NonLinearPerceptron import NonlinearPerceptron

# Accuracy calculation function
def accuracy(predictions, actual):
    correct = np.sum(predictions == actual)
    total = len(actual)
    return correct / total

# Traning
if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv('TP3-ej2-conjunto.csv')

    # Convert DataFrame to NumPy array
    X = df.to_numpy()

    # Separate the last column as the target labels (y) and the rest as features (X)
    y = X[:, -1]  # Get the last column (labels)
    X = X[:, :-1]  # Get all columns except the last one (features)

    # Add a column of ones to account for bias
    ones_column = np.ones((X.shape[0], 1))
    X = np.hstack((ones_column, X))

    # Normalize the input features to the range [-1, 1]
    epsilon = 1e-8
    X = 2 * (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + epsilon) - 1

    print("Target labels (y):", np.unique(y))

    # Convert continuous labels to binary (-1 and 1)
    median_value = np.median(y)
    y_binary = np.where(y > median_value, 1, -1)

    # Now y_binary contains the binary labels
    print("Binary target labels (y_binary):", np.unique(y_binary))  

    # Train the Linear Perceptron
    linear_perceptron = LinearPerceptron(learning_rate=0.025, max_iterations=10000)
    linear_trained_weights = linear_perceptron.fit(X, y_binary)
    print("Linear Perceptron - Trained weights:", linear_trained_weights)

    # Train the Nonlinear Perceptron
    nonlinear_perceptron = NonlinearPerceptron(learning_rate=0.025, max_iterations=10000)
    nonlinear_trained_weights = nonlinear_perceptron.fit(X, y_binary)
    print("Nonlinear Perceptron - Trained weights:", nonlinear_trained_weights)

    # Make predictions with both models
    linear_predictions = linear_perceptron.predict(X)
    nonlinear_predictions = nonlinear_perceptron.predict(X)

    # Calculate and print accuracy for both models
    linear_accuracy = accuracy(linear_predictions, y_binary)
    nonlinear_accuracy = accuracy(nonlinear_predictions, y_binary)
    
    print(f"Linear Perceptron Accuracy: {linear_accuracy}")
    print(f"Nonlinear Perceptron Accuracy: {nonlinear_accuracy}")