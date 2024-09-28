import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None

    def initialize_weights(self, n_features):
        # Initialize weights randomly in the range [-1, 1]
        self.weights = np.random.rand(n_features) * 2 - 1
        print(self.weights)

    def predict(self, x):
        # Calculate the activation, including the bias as part of weights
        activation = np.dot(x, self.weights)
        return np.sign(activation)

    def calculate_error(self, x, y):
        predictions = self.predict(x)
        return np.sum(predictions != y)

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.initialize_weights(n_features)

        error = 1
        min_error = n_samples * 2

        iteration = 0
        while error > 0 and iteration < self.max_iterations:
            ix = np.random.randint(0, n_samples)  # Randomly select an index
            excitation = np.dot(x[ix], self.weights)
            activation = np.sign(excitation)

            # Update weights
            delta_w = self.learning_rate * (y[ix] - activation) * x[ix]
            self.weights += delta_w

            # Calculate the current error
            error = self.calculate_error(x, y)

            # Update minimum error if needed
            min_error = min(min_error, error)

            iteration += 1

        return self.weights


# Example usage
if __name__ == "__main__":
    # Create a simple dataset for AND problem with a column of ones for bias
    X = np.array([[1, -1, -1],  # Adding bias as the first feature
                  [1, -1, 1],
                  [1, 1, -1],
                  [1, 1, 1]])  # Each row has a bias (1) and two features

    Y = np.array([-1, -1, -1, 1])  # Labels for the AND problem

    perceptron = Perceptron(learning_rate=0.1, max_iterations=1000)
    trained_weights = perceptron.fit(X, Y)

    print("Trained weights (including bias):", trained_weights)
