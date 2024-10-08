import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def initialize(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)

    def update(self, w, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        w_update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return w + w_update


import numpy as np

class Perceptron:
    def __init__(self, layer_sizes, beta=1, learning_rate=0.1, momentum=0):
        self.layer_sizes = layer_sizes
        self.beta = beta
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weights = None
        self.prev_weight_updates = None
        self.optimizers = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-(self.beta * x)))

    def sigmoid_derivative(self, x):
        sigmoid_val = self.sigmoid(x)
        return self.beta * sigmoid_val * (1 - sigmoid_val)

    def initialize_weights(self):
        self.weights = []
        self.optimizers = []
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.rand(self.layer_sizes[i], self.layer_sizes[i + 1]) * 2 - 1
            self.weights.append(w)
            optimizer = AdamOptimizer(learning_rate=self.learning_rate)
            optimizer.initialize(w.shape)
            self.optimizers.append(optimizer)
        self.prev_weight_updates = [np.zeros_like(w) for w in self.weights]

    def forward_propagation(self, x):
        activations = [x]
        excitations = []
        for i in range(len(self.weights)):
            net_input = np.dot(activations[-1], self.weights[i])
            excitations.append(net_input)
            activation = self.sigmoid(net_input)
            activations.append(activation)
        return activations, excitations

    def back_propagation(self, y_true, activations, excitations):
        errors = [None] * len(self.weights)
        weight_updates = [None] * len(self.weights)
        output_error = (y_true - activations[-1]) * self.sigmoid_derivative(excitations[-1])
        errors[-1] = output_error
        for i in reversed(range(len(self.weights) - 1)):
            errors[i] = np.dot(errors[i + 1], self.weights[i + 1].T) * self.sigmoid_derivative(excitations[i])
        for i in range(len(self.weights)):
            weight_updates[i] = np.dot(activations[i].T, errors[i])
        return weight_updates

    @staticmethod
    def compute_error(y_true, y_pred):
        return 0.5 * np.sum((y_true - y_pred) ** 2)

    def train(self, x, y, max_iterations, error_limit):
        self.initialize_weights()
        min_error = np.inf
        best_weights = None
        iteration = 0
        errors = []
        while min_error >= error_limit and iteration < max_iterations:
            sample_idx = np.random.randint(0, x.shape[0])
            x_sample = x[sample_idx:sample_idx + 1]
            y_sample = y[sample_idx:sample_idx + 1]
            activations, excitations = self.forward_propagation(x_sample)
            weight_updates = self.back_propagation(y_sample, activations, excitations)
            for i in range(len(self.weights)):
                self.weights[i] = self.optimizers[i].update(self.weights[i], weight_updates[i])
            predictions, _ = self.forward_propagation(x)
            error = self.compute_error(y, predictions[-1])
            errors.append(error)
            if error < min_error:
                min_error = error
                best_weights = [w.copy() for w in self.weights]
            iteration += 1
        return best_weights, min_error

    def predict(self, x):
        activations, _ = self.forward_propagation(x)
        return activations[-1]
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
