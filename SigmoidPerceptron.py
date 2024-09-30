import pandas as pd
import numpy as np

from Perceptron import Perceptron


class SigmoidPerceptron(Perceptron):
    def __init__(self, learning_rate=0.025, error_limit=0, max_iterations=10000):
        super().__init__(learning_rate, error_limit, max_iterations)

    @staticmethod
    def activate(excitations):
        return np.tanh(excitations)

    def calculate_error(self, x, y):
        predictions = self.predict(x)
        return 0.5 * np.sum((y - predictions) ** 2)  # Almost Sum of Squared Errors


# Example usage
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

    perceptron = SigmoidPerceptron(learning_rate=0.025, max_iterations=10000)
    trained_weights = perceptron.fit(X, Y)

    print("Trained weights (including bias):", trained_weights)
