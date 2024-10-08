import numpy as np


class Perceptron:
    def __init__(self, layer_sizes, beta=1, learning_rate=0.1, momentum=0):
        self.layer_sizes = layer_sizes  # List defining the number of neurons per layer
        self.beta = beta
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weights = None
        self.prev_weight_updates = None

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-(self.beta * x)))

    # Derivative of the sigmoid (calculated on weighted sums, not activations)
    def sigmoid_derivative(self, x):
        sigmoid_val = self.sigmoid(x)
        return self.beta * sigmoid_val * (1 - sigmoid_val)

    # Initialize the weights for all layers
    def initialize_weights(self):
        weights = []
        for i in range(len(self.layer_sizes) - 1):
            # Initialize weights randomly in the range [-1, 1]
            w = np.random.rand(self.layer_sizes[i], self.layer_sizes[i + 1]) * 2 - 1
            weights.append(w)
        print('INITIAL:', weights)
        self.weights = weights
        self.prev_weight_updates = [np.zeros_like(w) for w in self.weights]

    # Forward propagation with returning activations and excitations (weighted sums)
    def forward_propagation(self, x):
        activations = [x]  # Store activations for each layer
        excitations = []  # Store weighted sums for each layer (before activation)

        for i in range(len(self.weights)):
            net_input = np.dot(activations[-1], self.weights[i])  # Weighted sum (excitation)
            excitations.append(net_input)
            activation = self.sigmoid(net_input)  # Apply activation function (sigmoid)
            activations.append(activation)

        return activations, excitations

    # Backpropagation for multiple layers
    def back_propagation(self, y_true, activations, excitations):
        errors = [None] * len(self.weights)  # Initialize error list
        weight_updates = [None] * len(self.weights)

        # Error at the output layer (last layer)
        output_error = (y_true - activations[-1]) * self.sigmoid_derivative(excitations[-1])
        errors[-1] = output_error

        # Back-propagate error through hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            errors[i] = np.dot(errors[i + 1], self.weights[i + 1].T) * self.sigmoid_derivative(excitations[i])

        # Calculate weight updates
        for i in range(len(self.weights)):
            weight_updates[i] = self.learning_rate * np.dot(activations[i].T, errors[i]) + self.momentum * \
                                self.prev_weight_updates[i]
            self.prev_weight_updates[i] = weight_updates[i]

        return weight_updates

    # Sort of compute sum of squared errors
    @staticmethod
    def compute_error(y_true, y_pred):
        return 0.5 * np.sum((y_true - y_pred) ** 2)

    # Train the perceptron using stochastic gradient descent
    def train(self, x, y, max_iterations, error_limit):
        self.initialize_weights()
        min_error = np.inf  # Initialize minimum error
        best_weights = None  # To store the best weights
        iteration = 0

        while min_error >= error_limit and iteration < max_iterations:
            # Randomly select a sample for SGD
            sample_idx = np.random.randint(0, x.shape[0])
            x_sample = x[sample_idx:sample_idx + 1]
            y_sample = y[sample_idx:sample_idx + 1]

            # Forward pass
            activations, excitations = self.forward_propagation(x_sample)

            # Backpropagation and weight updates
            weight_updates = self.back_propagation(y_sample, activations, excitations)
            for i in range(len(self.weights)):
                self.weights[i] += weight_updates[i]

            # Compute error across the whole dataset
            predictions, _ = self.forward_propagation(x)
            error = self.compute_error(y, predictions[-1])

            # Update the minimum error and best weights if the current error is lower
            if error < min_error:
                min_error = error
                best_weights = [w.copy() for w in self.weights]  # Store the best weights

            iteration += 1

        print(iteration)
        return best_weights, min_error, iteration

    # Predict output for given input X
    def predict(self, x):
        activations, _ = self.forward_propagation(x)
        return activations[-1]  # Return the output from the last layer


# Example usage
if __name__ == "__main__":
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
    Y = np.array([[0], [1], [1], [0]])  # Expected outputs

    # Instantiate the Perceptron class
    mlp = Perceptron([2, 4, 1], beta=3, learning_rate=0.1)

    # Train the MLP
    trained_weights, err = mlp.train(X, Y, 100000, 0.1)

    print("Trained weights:", trained_weights)
    print("Minimum error:", err)

    # Testing the trained network on the XOR problem
    prediction = mlp.predict(X)
    print("Predictions:", prediction)
