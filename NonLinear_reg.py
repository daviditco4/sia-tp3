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

    def predict(self, X):
        # Compute the dot product of inputs and weights, followed by activation
        excitations = np.dot(X, self.weights)
        return self.activate(excitations)

    def fit(self, X, y, regularization_factor=0.01):
        n_samples, n_features = X.shape
        self.initialize_weights(n_features)
        
        iteration = 0
        
        while iteration < self.max_iterations:
            ix = np.random.randint(0, n_samples)  # Randomly select an index
            prediction = self.predict(X[ix])

            # Update weights using gradient descent for regression + regularization
            delta_w = self.learning_rate * (y[ix] - prediction) * X[ix] - regularization_factor * self.weights
            self.weights += delta_w

            # Calculate the current MSE
            error = self.calculate_error(X, y)
            if error <= self.error_limit:
                break  # Stop if error threshold is met
            iteration += 1

        print(f"Training completed in {iteration} iterations")
        return self.weights

    def calculate_error(self, X, y):
        predictions = self.predict(X)
        return np.mean((y - predictions) ** 2)  # Mean Squared Error


class NonlinearPerceptron(Perceptron):
    @staticmethod
    def activate(excitations):
        # Sigmoid activation function for continuous output
        excitations = np.clip(excitations, -300, 300)
        return 1 / (1 + np.exp(-excitations))


class LinearPerceptron(Perceptron):
    @staticmethod
    def activate(excitations):
        # Identity activation function for linear perceptron
        return excitations