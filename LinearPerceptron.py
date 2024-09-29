import pandas as pd
import numpy as np

from Perceptron import Perceptron


class LinearPerceptron(Perceptron):
    @staticmethod
    def activate(excitation):
        return excitation


if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv('exercise2/data/set.csv')

    # Convert DataFrame to NumPy array
    X = df.to_numpy()

    # Separate the last column and remove it from the matrix
    Y = X[:, -1]  # Get the last column
    X = X[:, :-1]  # Remove the last column

    # Create a column of ones
    ones_column = np.ones((X.shape[0], 1))

    # Concatenate the column of ones to the matrix
    X = np.hstack((ones_column, X))

    perceptron = LinearPerceptron(learning_rate=0.1, epsilon=0, max_iterations=1000)
    trained_weights = perceptron.fit(X, Y)

    print("Trained weights (including bias):", trained_weights)
