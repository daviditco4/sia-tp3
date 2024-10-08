import pandas as pd
import numpy as np

from Perceptron import Perceptron, min_max_normalize, min_max_normalize_output


class LinearPerceptron(Perceptron):
    def __init__(self, learning_rate=0.025, error_limit=0, max_iterations=10000):
        super().__init__(learning_rate, error_limit, max_iterations)

    @staticmethod
    def activate(excitations):
        return excitations

    def calculate_error(self, x, y):
        predictions = self.predict(x)
        return 0.5 * np.sum((y - predictions) ** 2)  # Almost Sum of Squared Errors


# Example usage
if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv('exercise2/data/set.csv')
    
    perceptron = LinearPerceptron(learning_rate=0.1, max_iterations=10000)
    
    # Convert DataFrame to NumPy array
    X = df.to_numpy()

    # Separate the last column and remove it from the matrix
    Y = X[:, -1]  # Get the last column
    X = X[:, :-1]  # Remove the last column

    #Normalize inputs and expected outputs
    norm_X, min_x, max_x = min_max_normalize(X)
    norm_Y, min_y, max_y = min_max_normalize_output(Y)
    
    print(norm_X)
    
    # Create a column of ones
    ones_column = np.ones((norm_X.shape[0], 1))
    # Concatenate the column of ones to the matrix
    norm_X = np.hstack((ones_column, norm_X))

    #Do the training
    trained_weights, iterations, errors = perceptron.fit(norm_X, norm_Y)
    new_errors = [float(i) for i in errors]

    print(new_errors)
    print(iterations)
    print("Trained weights (including bias):", trained_weights)
