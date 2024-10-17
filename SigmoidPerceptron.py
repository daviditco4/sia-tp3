import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold

from Perceptron import Perceptron, min_max_normalize, min_max_normalize_output

def getErrorAveragesandStd(learning_rate):
    test_errors=[]
    learning_error=[]
    for t in range(10):
        # Define the k-fold cross-validation strategy (e.g., 5 folds)
        kfold = KFold(n_splits=5, shuffle=True, random_state=random.randint(1,420))
        
        for train_index, test_index in kfold.split(norm_X):
            # Split data into training and testing sets
            X_train, X_test = norm_X[train_index], norm_X[test_index]
            y_train, y_test = norm_Y[train_index], norm_Y[test_index]  
            
            perceptron = SigmoidPerceptron(learning_rate=learning_rate, max_iterations=350)  
            trained_weights, iterations, errors = perceptron.fit(X_train, y_train)
            errors = [float(i) for i in errors]
            learning_error.append(errors)
            predict_error = perceptron.calculate_error(X_test, y_test)
            predict_error = float(predict_error)
            test_errors.append(predict_error)
            
    averages_list = [sum(elements) / len(elements) for elements in zip(*learning_error)] 
    std_devs = []
    for elements in zip(*learning_error):
        mean = sum(elements) / len(elements)
        variance = sum((x - mean) ** 2 for x in elements) / len(elements)
        std_devs.append(math.sqrt(variance))
    
    return averages_list, std_devs, test_errors

def getErrorAveragesandStd_v2(learning_rate):
    test_errors=[]
    learning_error=[]
    for mm in range(10):
        # Define the k-fold cross-validation strategy (e.g., 5 folds)
        kfold = KFold(n_splits=5, shuffle=True, random_state=random.randint(1, 420))
        
        for train_index, test_index in kfold.split(norm_X):
            # Split data into training and testing sets
            X_train, X_test = norm_X[train_index], norm_X[test_index]
            y_train, y_test = norm_Y[train_index], norm_Y[test_index]  
            
            perceptron = SigmoidPerceptron(learning_rate=learning_rate, max_iterations=1000)  
            errors, test_error_l = perceptron.fit_v2(X_train, y_train, X_test, y_test, min_x, max_x)
            
            errors = [float(i) for i in errors]
            test_error_l = [float(i) for i in test_error_l]
            
            learning_error.append(errors)
            test_errors.append(test_error_l)
            
    averages_list = [sum(elements) / len(elements) for elements in zip(*learning_error)] 
    test_averages_list = [sum(elements) / len(elements) for elements in zip(*test_errors)] 
    std_devs = []
    for elements in zip(*learning_error):
        mean = sum(elements) / len(elements)
        variance = sum((x - mean) ** 2 for x in elements) / len(elements)
        std_devs.append(math.sqrt(variance))
    
    std_devs_test = []
    for elements_t in zip(*test_errors):
        mean = sum(elements_t) / len(elements_t)
        variance = sum((k - mean) ** 2 for k in elements_t) / len(elements_t)
        std_devs_test.append(math.sqrt(variance))
    
    return averages_list, std_devs, test_averages_list, std_devs_test


class SigmoidPerceptron(Perceptron):
    def __init__(self, learning_rate=0.025, error_limit=0, max_iterations=10000):
        super().__init__(learning_rate, error_limit, max_iterations)

    @staticmethod
    def activate(excitations):
        return np.tanh(excitations)
    
    @staticmethod
    def derivative_activation(value):
        result_log = 1 / (1 + np.exp(-value))
        result_log_f = result_log * (1-result_log)
        result_log_f = result_log_f * 1
        result_tanh = 1 - (np.tanh(value) ** 2)
        result_tanh_f = result_tanh * 1
        return result_tanh_f

    def calculate_error(self, x, y):
        predictions = self.predict(x)
        return (1/len(predictions)) * np.sum((y - predictions) ** 2)  # Almost Sum of Squared Errors


# Example usage
if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv('exercise2/data/set.csv')

    # Convert DataFrame to NumPy array
    X = df.to_numpy()

    #Normalize inputs and expected outputs
    norm_X, min_x, max_x = min_max_normalize_output(X)
    print(norm_X)

    # Separate the last column and remove it from the matrix
    norm_Y = norm_X[:, -1]  # Get the last column
    norm_X = norm_X[:, :-1]  # Remove the last column

    # Create a column of ones
    ones_column = np.ones((norm_X.shape[0], 1))
    # Concatenate the column of ones to the matrix
    norm_X = np.hstack((ones_column, norm_X))
    
    avg_0_01, std_0_01, avg_test_0_01, std_test_0_01 = getErrorAveragesandStd_v2(0.01)
    avg_0_05, std_0_05, avg_test_0_05, std_test_0_05 = getErrorAveragesandStd_v2(0.05)
    avg_0_1, std_0_1, avg_test_0_1, std_test_0_1 = getErrorAveragesandStd_v2(0.1)
    avg_0_3, std_0_3, avg_test_0_3, std_test_0_3 = getErrorAveragesandStd_v2(0.3)
    avg_0_5, std_0_5, avg_test_0_5, std_test_0_5 = getErrorAveragesandStd_v2(0.5)

    
    # Plotting
    columns = range(1, len(avg_0_5) + 1)  # Column indices (1 to 200)
    plt.figure(figsize=(12, 6))

    # Plot the averages and standard deviation
    plt.plot(columns, avg_0_01, marker='o', color='fuchsia', label='Learning Rate 20 Training', linestyle='-')
    plt.errorbar(columns, avg_0_01, yerr=std_0_01, fmt='o', color='fuchsia', capsize=5)

    plt.plot(columns, avg_test_0_01, marker='o', color='grey', label='Learning Rate 0.01 Test', linestyle='-')
    plt.errorbar(columns, avg_test_0_01, yerr=std_test_0_01, fmt='o', color='grey', capsize=5)

    plt.plot(columns, avg_0_05, marker='o', color='gold', label='Learning Rate 22.5 Training', linestyle='-')
    plt.errorbar(columns, avg_0_05, yerr=std_0_05, fmt='o', color='gold', capsize=5)
    
    plt.plot(columns, avg_test_0_05, marker='o', color='indigo', label='Learning Rate 0.05 Test', linestyle='-')
    plt.errorbar(columns, avg_test_0_05, yerr=std_test_0_05, fmt='o', color='indigo', capsize=5)

    plt.plot(columns, avg_0_1, marker='o', color='skyblue', label='Learning Rate 25 Training', linestyle='-')
    plt.errorbar(columns, avg_0_1, yerr=std_0_1, fmt='o', color='skyblue', capsize=5)
    
    plt.plot(columns, avg_test_0_1, marker='o', color='orange', label='Learning Rate 0.1 Test', linestyle='-')
    plt.errorbar(columns, avg_test_0_1, yerr=std_test_0_1, fmt='o', color='orange', capsize=5)

    plt.plot(columns, avg_0_3, marker='o', color='orchid', label='Learning Rate 27.5 Training', linestyle='-')
    plt.errorbar(columns, avg_0_3, yerr=std_0_3, fmt='o', color='orchid', capsize=5)
    
    plt.plot(columns, avg_test_0_3, marker='o', color='goldenrod', label='Learning Rate 0.3 Test', linestyle='-')
    plt.errorbar(columns, avg_test_0_3, yerr=std_test_0_3, fmt='o', color='goldenrod', capsize=5)

    plt.plot(columns, avg_0_5, marker='o', color='skyblue', label='Learning Rate 30 Training', linestyle='-')
    plt.errorbar(columns, avg_0_5, yerr=std_0_5, fmt='o', color='skyblue', capsize=5)
    
    plt.plot(columns, avg_test_0_5, marker='o', color='orchid', label='Learning Rate 30 Test', linestyle='-')
    plt.errorbar(columns, avg_test_0_5, yerr=std_test_0_5, fmt='o', color='orchid', capsize=5)

    # Add labels and title
    plt.xticks(ticks=range(0, 1001, 50))  # Set x-ticks to the column numbers
    plt.xlabel('Iteration of Perceptron')
    plt.ylabel('Average Error')
    plt.title('Average training error vs test error per iteration of Non-Linear Perceptron with Sigmoid')
    plt.grid(axis='y')  # Add grid lines for better readability
    plt.legend()  # Show legend
    plt.show()
