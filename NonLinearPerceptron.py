import pandas as pd
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.025, error_limit=0, max_iterations=10000):
        self.learning_rate = learning_rate
        self.error_limit = error_limit
        self.max_iterations = max_iterations
        self.weights = None

    def initialize_weights(self, n_features):
        # Initialize weights randomly in the range [-1, 1]
        self.weights = np.random.rand(n_features) * 2 - 1
        print("Initial weights:", self.weights)

    #def predict(self, X):
        # Compute the dot product of inputs and weights
     #   excitations = np.dot(X, self.weights)
     #   return self.activate(excitations)
    
    def predict(self, x):
        # Calculate the excitations, including the bias as part of weights
        excitations = np.dot(x, self.weights)
    
    # Apply the sigmoid activation function
        activations = self.activate(excitations)
    
    # Apply threshold: return 1 for activations > 0.5, -1 for activations <= 0.5
        return np.where(activations > 0.5, 1, -1)

    def fit(self, X, y, regularization_factor=0.01):
        n_samples, n_features = X.shape
        self.initialize_weights(n_features)
        
        min_error = np.inf
        iteration = 0
        
        while min_error >= self.error_limit and iteration < self.max_iterations:
            ix = np.random.randint(0, n_samples)  # Randomly select an index
            activation = self.predict(X[ix])

            # Update weights using the perceptron learning rule + regularization
            delta_w = self.learning_rate * (y[ix] - activation) * X[ix] - regularization_factor * self.weights
            
            # Clip the weight updates to avoid exploding weights
            delta_w = np.clip(delta_w, -1, 1)
            self.weights += delta_w

            # Calculate the current error
            error = self.calculate_error(X, y)

            # Update minimum error if needed
            min_error = min(min_error, error)
            iteration += 1

        print(f"Training completed in {iteration} iterations")
        return self.weights

    def calculate_error(self, X, y):
        predictions = self.predict(X)
        return 0.5 * np.sum((y - predictions) ** 2)  # Sum of Squared Errors


class NonlinearPerceptron(Perceptron):
    @staticmethod
    def activate(excitations):
        excitations = np.clip(excitations, -300, 300)  # Adjust the range if necessary
        return 1 / (1 + np.exp(-excitations))

    def calculate_error(self, X, y):
        predictions = self.predict(X)
        return 0.5 * np.sum((y - predictions) ** 2)  # Squared error as in the linear case


# Example usage
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

    # Normalize the input features to the range [-1, 1], avoiding division by zero
    epsilon = 1e-8
    X = 2 * (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + epsilon) - 1
   
    # Initialize the Nonlinear Perceptron and train it
    perceptron = NonlinearPerceptron(learning_rate=0.025, max_iterations=10000)
    trained_weights = perceptron.fit(X, y)

    print("Trained weights (including bias):", trained_weights)