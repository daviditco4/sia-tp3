import pandas as pd
import numpy as np
from LinearPerceptron import LinearPerceptron
from NonLinearPerceptron import NonlinearPerceptron
from sklearn.model_selection import train_test_split

# Accuracy calculation function
def accuracy(predictions, actual):
    correct = np.sum(predictions == actual)
    total = len(actual)
    return correct / total

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

    # Split the dataset into training and test sets (80% training, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    ## -- Train and Evaluate on Training Set --

    # Train the Linear Perceptron on training data
    linear_perceptron = LinearPerceptron(learning_rate=0.025, max_iterations=10000)
    linear_trained_weights = linear_perceptron.fit(X_train, y_train)
    print("Linear Perceptron - Trained weights:", linear_trained_weights)

    # Train the Nonlinear Perceptron on training data
    nonlinear_perceptron = NonlinearPerceptron(learning_rate=0.025, max_iterations=10000)
    nonlinear_trained_weights = nonlinear_perceptron.fit(X_train, y_train)
    print("Nonlinear Perceptron - Trained weights:", nonlinear_trained_weights)

    ## -- Make Predictions and Calculate Training Accuracy --

    # Make predictions on the training data
    linear_train_predictions = linear_perceptron.predict(X_train)
    nonlinear_train_predictions = nonlinear_perceptron.predict(X_train)

    # Calculate and print training accuracy for both models
    linear_train_accuracy = accuracy(linear_train_predictions, y_train)
    nonlinear_train_accuracy = accuracy(nonlinear_train_predictions, y_train)
    
    print(f"Linear Perceptron Training Accuracy: {linear_train_accuracy}")
    print(f"Nonlinear Perceptron Training Accuracy: {nonlinear_train_accuracy}")

    ## -- Evaluate on Test Set --

    # Make predictions on the test data
    linear_test_predictions = linear_perceptron.predict(X_test)
    nonlinear_test_predictions = nonlinear_perceptron.predict(X_test)

    # Calculate and print test accuracy for both models
    linear_test_accuracy = accuracy(linear_test_predictions, y_test)
    nonlinear_test_accuracy = accuracy(nonlinear_test_predictions, y_test)
    
    print(f"Linear Perceptron Test Accuracy: {linear_test_accuracy}")
    print(f"Nonlinear Perceptron Test Accuracy: {nonlinear_test_accuracy}")