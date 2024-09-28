from ast import literal_eval

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function to plot the decision boundary
def plot_decision_boundary(weights):
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5

    # Create a grid of points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Calculate the decision boundary
    z = weights[0] + weights[1] * xx + weights[2] * yy  # weights[0] is bias
    plt.contourf(xx, yy, z, levels=[-10, 0, 10], alpha=0.2, colors=('red', 'blue'))


# Function to load weights from a CSV file
def load_weights(csv_file):
    data = pd.read_csv(csv_file, sep=';', converters=dict(weights=literal_eval))
    weights = data['weights'].values[3]
    print(weights)
    return weights


# Main function
def main():
    # Load weights from CSV file
    csv_file = 'exercise1/outputs/and_weights.csv'  # Update with your CSV file path
    weights = load_weights(csv_file)

    # AND problem dataset
    x = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    y = np.array([-1, -1, -1, 1])  # Labels for the AND problem

    # Plot the dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], color='blue', label='Class -1 (0)', s=100)
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color='red', label='Class 1 (1)', s=100)

    # Plot decision boundary
    plot_decision_boundary(weights)

    # Add labels and legend
    plt.title('AND Problem with Perceptron Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
