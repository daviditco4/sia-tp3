import pandas as pd
import numpy as np
from LinearPerceptron import LinearPerceptron
from NonLinearPerceptron import NonlinearPerceptron

# Read the CSV file and prepare the data
df = pd.read_csv('TP3-ej2-conjunto.csv')
X = df.to_numpy()
y = X[:, -1]  # Last column as target values
X = X[:, :-1]  # All columns except last as features

# Normalize y to range [0, 1]
y_min = y.min()
y_max = y.max()
y = (y - y_min) / (y_max - y_min)


# Mean Squared Error calculation function
def mean_squared_error(predictions, actual):
    return np.mean((predictions - actual) ** 2)

# Training for regression
if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv('TP3-ej2-conjunto.csv')

    # Convert DataFrame to NumPy array
    X = df.to_numpy()

    # Separate the last column as the target values (y) and the rest as features (X)
    y = X[:, -1]  # Get the last column (target values)
    X = X[:, :-1]  # Get all columns except the last one (features)

    # Add a column of ones to account for bias
    ones_column = np.ones((X.shape[0], 1))
    X = np.hstack((ones_column, X))

    # Normalize the input features to the range [-1, 1]
    epsilon = 1e-8
    X = 2 * (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + epsilon) - 1

    # Split the dataset into training and test sets (80% training, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Nonlinear Perceptron on training data
    nonlinear_perceptron = NonlinearPerceptron(learning_rate=0.025, max_iterations=10000)
    nonlinear_trained_weights = nonlinear_perceptron.fit(X_train, y_train)
    print("Nonlinear Perceptron - Trained weights:", nonlinear_trained_weights)

    # Make predictions and calculate Mean Squared Error on the training and test data
    train_predictions = nonlinear_perceptron.predict(X_train)
    test_predictions = nonlinear_perceptron.predict(X_test)
    
    train_mse = mean_squared_error(train_predictions, y_train)
    test_mse = mean_squared_error(test_predictions, y_test)
    
    print(f"Nonlinear Perceptron Training Mean Squared Error: {train_mse}")
    print(f"Nonlinear Perceptron Test Mean Squared Error: {test_mse}")