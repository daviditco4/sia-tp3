import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error

def min_max_normalize_sigmoid(X):
        X_min = np.min(X)
        X_max = np.max(X)
        epsilon = 1e-8
        return ((X - X_min) / (X_max - X_min + epsilon)), X_min, X_max
    
def min_max_normalize_output(y):
    y_min = np.min(y)
    y_max = np.max(y)
    epsilon = 1e-8
    return (2 * ((y - y_min) / (y_max - y_min + epsilon))) - 1, y_min, y_max

# Function to denormalize output back to original scale (useful for regression tasks)
def denormalize_output(y_norm, y_min, y_max):
    return (((y_norm + 1)/2) * (y_max - y_min + 1e-8)) + y_min

def denormalize_output_sigmoid(y_norm, y_min, y_max):
    return ((y_norm) * (y_max - y_min + 1e-8)) + y_min

def meanSquareError(y_true, y_pred):
    error_sum = 0
    for i in range(len(y_true)):
        aux = y_true[i] - y_pred[i]
        error_sum = error_sum + (aux ** 2)
    final_error = (error_sum / len(y_true))
    return final_error

#task1
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

    @staticmethod
    def derivative_activation(value):
        return 1

    #def predict(self, x):
        # Calculate the excitations, including the bias as part of weights
     #   excitations = np.dot(x, self.weights)
     #   return self.activate(excitations)
    
    def predict(self, x):
    # Calculate the excitations, including the bias as part of weights
        excitations = np.dot(x, self.weights)
        return excitations

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
            for ix in range(len(x)):
                #Weigthed sum of inputs and weights
                weighted_sum = self.predict(x[ix])
                
                #Activation of the weighted sum
                activation = self.activate(weighted_sum)

                # Update weights
                delta_w = self.learning_rate * (y[ix] - activation) * x[ix] * self.derivative_activation(weighted_sum)
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
            
            #Predict with test data and give error

        return self.weights, iteration, errors

    def fit_v2(self, x, y, test_x, test_y, min_a, max_a):
        n_samples, n_features = x.shape
        self.initialize_weights(n_features)

        tr_errors = []
        tt_errors = []
        min_error = np.inf

        iteration = 0
        while min_error >= self.error_limit and iteration < self.max_iterations:
            training_set_errors = []
            for ix in range(len(x)):
                #Weigthed sum of inputs and weights
                weighted_sum = self.predict(x[ix])
                
                #Activation of the weighted sum
                activation = self.activate(weighted_sum)

                # Update weights
                delta_w = self.learning_rate * (y[ix] - activation) * x[ix] * self.derivative_activation(weighted_sum)
                self.weights += delta_w
                
                #Calculating error of this training example
                #weighted_sum2 = self.predict(x[ix])
                #y_pred = self.activate(weighted_sum2)
                
                #De-normalize predicted and expected output
                #denorm_y = denormalize_output(y[ix], min_a, max_a)
                #denorm_y_pred = denormalize_output(y_pred, min_a, max_a)
                
                # Calculate the error of this training example
                #error = (denorm_y - denorm_y_pred) ** 2
                #training_set_errors.append(error)
                
            iteration += 1
            
            #avg_ts_error = (sum(training_set_errors) / len(training_set_errors))
            #tr_errors.append(avg_ts_error)
            
            weigthed_sum_full_tr = self.predict(x)
            train_y_predict = self.activate(weigthed_sum_full_tr)
            
            denorm_Y_pred = denormalize_output(train_y_predict, min_a, max_a)
            denorm_Y = denormalize_output(y, min_a, max_a)
            
            y_train_error = root_mean_squared_error(denorm_Y, denorm_Y_pred)
            tr_errors.append(y_train_error)
            
            #Predict with test data and give error
            weighted_sum_test = self.predict(test_x)
            y_pred_test = self.activate(weighted_sum_test)
            
            #De-normalize predicted and expected output
            denorm_y_t = denormalize_output(test_y, min_a, max_a)
            denorm_y_pred_t = denormalize_output(y_pred_test, min_a, max_a)
            
            y_test_error = root_mean_squared_error(denorm_y_t, denorm_y_pred_t)
            tt_errors.append(y_test_error)

        return tr_errors, tt_errors



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
    
    
    
