import numpy as np

#task1
class Perceptron:
    def __init__(self, learning_rate=0.1, error_limit=0.2, max_iterations=1000):
        self.learning_rate = learning_rate
        self.error_limit = error_limit
        self.max_iterations = max_iterations
        self.weights = None

    def initialize_weights(self, n_features):
        # Initialize weights randomly in the range [-1, 1]
        self.weights = np.random.rand(n_features) * 2 - 1
        print(self.weights)

    @staticmethod
    def activate(excitations):
        return np.where(excitations == 0, 1, np.sign(excitations))

    #def predict(self, x):
        # Calculate the excitations, including the bias as part of weights
     #   excitations = np.dot(x, self.weights)
     #   return self.activate(excitations)
    
    def predict(self, x):
    # Calculate the excitations, including the bias as part of weights
        excitations = np.dot(x, self.weights)
    
    # Apply threshold: return 1 for positive excitations, -1 for negative or zero
        return np.where(excitations > 0, 1, -1)

    def calculate_error(self, x, y):
        predictions = self.predict(x)
        return np.sum(predictions != y)

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.initialize_weights(n_features)

        min_error = np.inf

        iteration = 0
        while min_error >= self.error_limit and iteration < self.max_iterations:
            ix = np.random.randint(0, n_samples)  # Randomly select an index
            activation = self.predict(x[ix])

            # Update weights
            delta_w = self.learning_rate * (y[ix] - activation) * x[ix]
            self.weights += delta_w

            # Calculate the current error
            error = self.calculate_error(x, y)

            # Update minimum error if needed
            min_error = min(min_error, error)

            iteration += 1

        print(iteration)
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
