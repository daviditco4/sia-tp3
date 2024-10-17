import numpy as np

from MultilayerPerceptron import Perceptron


class AdamMultilayerPerceptron(Perceptron):
    def __init__(self, layer_sizes, beta=1.0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_updates_by_epoch=False):
        super().__init__(layer_sizes, beta, learning_rate, 0.0, weight_updates_by_epoch)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 1  # Time step for bias correction
        self.m_weights = None
        self.v_weights = None

    def initialize_weights(self):
        super().initialize_weights()
        # Initialize Adam parameters (correction, moving averages of gradients and squared gradients)
        self.t = 1
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]

    def update_weights(self, weight_gradients):
        weight_updates = [None] * len(self.weights)

        for i in range(len(self.weights)):
            # Update biased first moment estimate (m) and second moment estimate (v) for weights
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * weight_gradients[i]
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (weight_gradients[i] ** 2)

            # Bias correction for weights
            m_weights_corr = self.m_weights[i] / (1 - self.beta1 ** self.t)
            v_weights_corr = self.v_weights[i] / (1 - self.beta2 ** self.t)

            # Update weights using Adam update rule
            weight_updates[i] = -self.learning_rate * m_weights_corr / (np.sqrt(v_weights_corr) + self.epsilon)

        self.t += 1
        return weight_updates


# Example usage
if __name__ == "__main__":
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Inputs
    Y = np.array([[0], [1], [1], [0]])  # Expected outputs

    # Instantiate the Perceptron class
    mlp = AdamMultilayerPerceptron([2, 4, 1], learning_rate=0.001, weight_updates_by_epoch=False)

    # Train the MLP
    trained_weights, err, epochs = mlp.train(X, Y, np.inf, 0.005)

    print("Trained weights:", trained_weights)
    print("Minimum error:", err)
    print("Epoch reached:", epochs)

    # Testing the trained network on the XOR problem
    prediction = mlp.predict(X)
    print("Predictions:", prediction)
