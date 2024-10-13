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
            
            perceptron = LinearPerceptron(learning_rate=learning_rate, max_iterations=350)  
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

class LinearPerceptron(Perceptron):
    def __init__(self, learning_rate=0.025, error_limit=0, max_iterations=10000):
        super().__init__(learning_rate, error_limit, max_iterations)

    @staticmethod
    def activate(excitations):
        return excitations

    def calculate_error(self, x, y):
        predictions = self.predict(x)
        return (1/len(predictions)) * np.sum((y - predictions) ** 2)  # Almost Sum of Squared Errors


# Example usage
if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv('exercise2/data/set.csv')
    
    # Convert DataFrame to NumPy array
    X = df.to_numpy()

    # Separate the last column and remove it from the matrix
    Y = X[:, -1]  # Get the last column
    X = X[:, :-1]  # Remove the last column

    #Normalize inputs and expected outputs
    norm_X, min_x, max_x = min_max_normalize(X)
    norm_Y, min_y, max_y = min_max_normalize_output(Y)
    print(norm_X)
    
    # Create a column of ones
    ones_column = np.ones((norm_X.shape[0], 1))
    # Concatenate the column of ones to the matrix
    norm_X = np.hstack((ones_column, norm_X))
    
    avg_0_1, std_0_1, test_0_1 = getErrorAveragesandStd(0.1)
    avg_0_01, std_0_01, test_0_01 = getErrorAveragesandStd(0.01)
    avg_0_25, std_0_25, test_0_25 = getErrorAveragesandStd(0.25)
    avg_0_4, std_0_4, test_0_4 = getErrorAveragesandStd(0.4)
    avg_0_5, std_0_5, test_0_5 = getErrorAveragesandStd(0.5)
    avg_0_75, std_0_75, test_0_75 = getErrorAveragesandStd(0.75)
    avg_0_9, std_0_9, test_0_9 = getErrorAveragesandStd(0.9)
    
    tests = []
    tests.append(test_0_01)
    tests.append(test_0_1)
    tests.append(test_0_25)
    tests.append(test_0_4)
    tests.append(test_0_5)
    tests.append(test_0_75)
    tests.append(test_0_9)
    
    # Plotting
    columns = range(1, len(avg_0_1) + 1)  # Column indices (1 to 200)
    plt.figure(figsize=(12, 6))

    # Plot the averages and standard deviation
    plt.plot(columns, avg_0_1, marker='o', color='skyblue', label='Learning Rate 0.1', linestyle='-')
    plt.errorbar(columns, avg_0_1, yerr=std_0_1, fmt='o', color='skyblue', capsize=5)
    
    plt.plot(columns, avg_0_01, marker='o', color='orange', label='Learning Rate 0.01', linestyle='-')
    plt.errorbar(columns, avg_0_01, yerr=std_0_01, fmt='o', color='orange', capsize=5)
    
    plt.plot(columns, avg_0_25, marker='o', color='orchid', label='Learning Rate 0.25', linestyle='-')
    plt.errorbar(columns, avg_0_25, yerr=std_0_25, fmt='o', color='orchid', capsize=5)

    plt.plot(columns, avg_0_4, marker='o', color='goldenrod', label='Learning Rate 0.4', linestyle='-')
    plt.errorbar(columns, avg_0_4, yerr=std_0_4, fmt='o', color='goldenrod', capsize=5)

    plt.plot(columns, avg_0_5, marker='o', color='mediumseagreen', label='Learning Rate 0.5', linestyle='-')
    plt.errorbar(columns, avg_0_5, yerr=std_0_5, fmt='o', color='mediumseagreen', capsize=5)

    plt.plot(columns, avg_0_75, marker='o', color='tomato', label='Learning Rate 0.75', linestyle='-')
    plt.errorbar(columns, avg_0_75, yerr=std_0_75, fmt='o', color='tomato', capsize=5)

    #plt.plot(columns, avg_0_9, marker='o', color='orchid', label='Learning Rate 0.9', linestyle='-')
    #plt.errorbar(columns, avg_0_9, yerr=std_0_9, fmt='o', color='orchid', capsize=5)

    # Add labels and title
    plt.xticks(ticks=range(0, 351, 10))  # Set x-ticks to the column numbers
    plt.xlabel('Iteration of Perceptron')
    plt.ylabel('Average Error')
    plt.title('Average error per iteration of Linear Perceptron with different learning rates')
    plt.grid(axis='y')  # Add grid lines for better readability
    plt.legend()  # Show legend
    plt.show()


    # Prepare the data for the DataFrame
    data = {
        'Min Value': [np.min(arr) for arr in tests],
        'Max Value': [np.max(arr) for arr in tests],
        'Average': [np.mean(arr) for arr in tests],
        'Standard Deviation': [np.std(arr) for arr in tests]
    }
    
    df = pd.DataFrame(data, index=['Learning Rate 0.01', 'Learning Rate 0.1', 'Learning Rate 0.25', 'Learning Rate 0.4', 'Learning Rate 0.5', 'Learning Rate 0.75', 'Learning Rate 0.9'])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create a table from the DataFrame
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')

    # Set the font size and other aesthetics
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.show()