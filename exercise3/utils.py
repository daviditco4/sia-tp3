import json
import os
import sys

import numpy as np

# Add the path to the folder containing MultilayerPerceptron.py
sys.path.append(os.path.abspath("."))

# Now you can import MultilayerPerceptron
from MultilayerPerceptron import Perceptron


def read_digits_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the 5x7 bitmaps for each digit
    bitmaps = []
    for i in range(0, len(lines), 7):
        bitmap = []
        for j in range(7):
            bitmap.extend([int(bit) for bit in lines[i + j].strip().split()])
        bitmaps.append(bitmap)

    return np.array(bitmaps)


def read_hyperparameters_from_json(file_path):
    with open(file_path, 'r') as file:
        hyperparameters = json.load(file)
    return hyperparameters


def train_perceptron(digits, labels, hyperparameters):
    # Initialize perceptron using the hyperparameters
    mlp = Perceptron(hyperparameters["layer_sizes"], beta=hyperparameters["beta"],
                     learning_rate=hyperparameters["learning_rate"],
                     momentum=hyperparameters["momentum"] if 'momentum' in hyperparameters else 0)

    # Train the perceptron
    best_weights, min_error, iterations = mlp.train(digits, labels, epoch_limit=50000,
                                                    error_limit=hyperparameters["error_limit"])

    print(min_error)
    return mlp, iterations
