import random

import numpy as np


class Perceptron:
    def __init__(self, layer_sizes, beta=1.0, learning_rate=0.001, momentum=0.0, weight_updates_by_epoch=False):
        self.layer_sizes = layer_sizes  # List defining the number of neurons per layer
        self.beta = beta
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_updates_by_epoch = weight_updates_by_epoch
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
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) / 3
            weights.append(w)
        self.weights = weights
        self.prev_weight_updates = [np.zeros_like(w) for w in self.weights]

    # Forward propagation with returning activations and excitations (weighted sums)
    def forward_propagation(self, x, weights=None):
        if weights is None:
            weights = self.weights
        activations = [x]  # Store activations for each layer
        excitations = []  # Store weighted sums for each layer (before activation)

        for i in range(len(weights)):
            net_input = np.dot(activations[-1], weights[i])  # Weighted sum (excitation)
            excitations.append(net_input)
            activation = self.sigmoid(net_input)  # Apply activation function (sigmoid)
            activations.append(activation)

        return activations, excitations

    def update_weights(self, weight_gradients):
        weight_updates = [None] * len(self.weights)

        for i in range(len(self.weights)):
            weight_updates[i] = -self.learning_rate * weight_gradients[i] + self.momentum * self.prev_weight_updates[i]
            self.prev_weight_updates[i] = weight_updates[i]

        return weight_updates

    # Backpropagation for multiple layers
    def back_propagation(self, y_true, activations, excitations):
        errors = [None] * len(self.weights)  # Initialize error list
        weight_gradients = [None] * len(self.weights)

        # Error at the output layer (last layer)
        output_error = (y_true - activations[-1]) * self.sigmoid_derivative(excitations[-1])
        errors[-1] = output_error

        # Back-propagate error through hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            errors[i] = np.dot(errors[i + 1], self.weights[i + 1].T) * self.sigmoid_derivative(excitations[i])

        # Calculate weight gradients
        for i in range(len(self.weights)):
            weight_gradients[i] = -np.dot(activations[i].T,
                                          errors[i])  # TODO: Evaluate if the assignment should be negative

        return weight_gradients

    # Compute mean squared error
    @staticmethod
    def compute_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # Train the perceptron using gradient descent
    def train(self, x, y, epoch_limit, error_limit):
        self.initialize_weights()
        weight_updates = [np.zeros_like(w) for w in self.weights]
        min_error = np.inf  # Initialize minimum error
        best_weights = None  # To store the best weights
        training_done = False
        epoch = 0
        weight_history = []
        error_history = []

        while not training_done and epoch < epoch_limit:
            indexes_shuffled = list(range(len(x)))
            random.shuffle(indexes_shuffled)

            for sample_idx in indexes_shuffled:
                x_sample = x[sample_idx:sample_idx + 1]
                y_sample = y[sample_idx:sample_idx + 1]

                # Forward pass
                activations, excitations = self.forward_propagation(x_sample)

                # Backpropagation and weight updates
                weight_gradients = self.back_propagation(y_sample, activations, excitations)
                weight_updates_aux = self.update_weights(weight_gradients)
                for i in range(len(self.weights)):
                    if not self.weight_updates_by_epoch:
                        self.weights[i] += weight_updates_aux[i]
                    else:
                        weight_updates[i] += weight_updates_aux[i]

                if not self.weight_updates_by_epoch:
                    # Compute error across the whole dataset
                    predictions, _ = self.forward_propagation(x)
                    error = self.compute_error(y, predictions[-1])

                    # Update the minimum error and best weights if the current error is lower
                    if error < min_error:
                        min_error = error
                        best_weights = [w.copy() for w in self.weights]  # Store the best weights
                        if min_error < error_limit:
                            training_done = True
                            break

            # Weight updates
            if self.weight_updates_by_epoch:
                for i in range(len(self.weights)):
                    self.weights[i] += weight_updates[i]
                weight_updates = [0] * len(self.weights)

            # Compute error across the whole dataset
            predictions, _ = self.forward_propagation(x)
            error = self.compute_error(y, predictions[-1])

            epoch += 1
            weight_history.append([w + wu for w, wu in zip(self.weights, weight_updates)])
            error_history.append(error)

            if self.weight_updates_by_epoch:
                # Update the minimum error and best weights if the current error is lower
                if error < min_error:
                    min_error = error
                    best_weights = [w.copy() for w in self.weights]  # Store the best weights
                    if min_error < error_limit:
                        break

        return best_weights, min_error, epoch, weight_history, error_history


    # Predict output for given input X
    def predict(self, x, weights=None):
        activations, _ = self.forward_propagation(x, weights)
        return activations[-1]  # Return the output from the last layer


# Example usage
if __name__ == "__main__":
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
    Y = np.array([[0], [1], [1], [0]])  # Expected outputs

    # Instantiate the Perceptron class
    mlp = Perceptron([2, 4, 1], learning_rate=0.6, momentum=0.9, weight_updates_by_epoch=False)

    # Train the MLP
    trained_weights, err, epochs = mlp.train(X, Y, np.inf, 0.005)

    print("Trained weights:", trained_weights)
    print("Minimum error:", err)
    print("Epoch reached:", epochs)

    # Testing the trained network on the XOR problem
    prediction = mlp.predict(X)
    print("Predictions:", prediction)
