import numpy as np
import matplotlib.pyplot as plt

def min_max_normalize(X):
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        epsilon = 1e-8
        return 2*((X - X_min) / (X_max - X_min + epsilon))-1, X_min, X_max
    
# Function to normalize output data (optional, but typically binary 0/1)
def min_max_normalize_output(y):
    y_min = np.min(y)
    y_max = np.max(y)
    epsilon = 1e-8
    return 2*((y - y_min) / (y_max - y_min + epsilon))-1, y_min, y_max

# Function to denormalize output back to original scale (useful for regression tasks)
def denormalize_output(y_norm, y_min, y_max):
    return ((y_norm + 1)/2) * (y_max - y_min + 1e-8) + y_min


class Perceptron:
    def __init__(self, learning_rate=0.1, error_limit=1, max_iterations=1000):
        self.learning_rate = learning_rate
        self.error_limit = error_limit
        self.max_iterations = max_iterations
        self.weights = None

    def initialize_weights(self, n_features):
        # Initialize weights randomly in the range [-1, 1]
        self.weights = np.random.rand(n_features) * 2 - 1
        #print(self.weights)

    @staticmethod
    def activate(excitations):
        return np.where(excitations == 0, 1, np.sign(excitations))

    def predict(self, x):
        # Calculate the excitations, including the bias as part of weights
        excitations = np.dot(x, self.weights)
        return self.activate(excitations)

    def calculate_error(self, x, y):
        predictions = self.predict(x)
        return np.sum(predictions != y)
    

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.initialize_weights(n_features)

        errors = []
        min_error = np.inf

        iteration = 0
        while min_error >= self.error_limit and iteration < self.max_iterations:
            ix = np.random.randint(0, n_samples)  # Randomly select an index
            activation = self.predict(x[ix])

            # Update weights
            delta_w = self.learning_rate * (y[ix] - activation) * x[ix]
            self.weights += delta_w
            
            #De-normalize inputs and expected output
            denorm_y = denormalize_output(y, np.min(y), np.max(y))
            denorm_x = denormalize_output(x, np.min(x), np.max(x))
            # Calculate the current error
            error = self.calculate_error(denorm_x, denorm_y)
            errors.append(error)

            # Update minimum error if needed
            min_error = min(min_error, error)

            iteration += 1

        return self.weights, iteration, errors


# Example usage
if __name__ == "__main__":
    # Create a simple dataset for AND problem with a column of ones for bias
    X = np.array([[1, -1, -1],  # Adding bias as the first feature
                  [1, -1, 1],
                  [1, 1, -1],
                  [1, 1, 1]])  # Each row has a bias (1) and two features

    Y = np.array([-1, 1, 1, -1])  # Labels for the XOR problem
    data = []
    for i in range(9):
        iteration = 0
        iterations = []
        
        perceptron = Perceptron(learning_rate=(i+1)/10, max_iterations=1000)
        
        for i in range(25):
            trained_weights, iteration = perceptron.fit(X, Y)
            iterations.append(iteration)
            #print("Trained weights (including bias):", trained_weights)
            
        data.append(iterations)
        print(i+1)
    
    # Calculate averages and standard deviations for each array
    means = [np.mean(arr) for arr in data]
    std_devs = [np.std(arr, ddof=1) for arr in data]  # Sample standard deviation
    
    # Plotting
    fig, ax = plt.subplots()

    # Bar positions
    x_pos = np.arange(len(data))

    # Create bar chart with error bars representing the standard deviation
    ax.bar(x_pos, means, yerr=std_devs, capsize=5, color='skyblue', edgecolor='black')

    # Labels and title
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Average Value of iterations')
    ax.set_title('Average Iterations by learning rate')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{(i+1)/10}' for i in range(len(data))])  # Label each dataset

    # Show the plot
    plt.tight_layout()
    plt.show()
    
    
    
