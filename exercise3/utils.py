import csv
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
        hyperparams = json.load(file)
    return hyperparams


def train_perceptron(x, y, hyperparams):
    # Initialize perceptron using the hyperparameters
    p = Perceptron(layer_sizes=hyperparams["layer_sizes"], beta=hyperparams["beta"],
                   learning_rate=hyperparams["learning_rate"])

    # Train the perceptron
    best_weights, min_error = p.train(x, y, max_iterations=np.inf, error_limit=hyperparams["error_limit"])

    print(min_error)
    return p


def append_results_to_csv(file_path, elap_time, hyperparams, accu):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # If the file is new, write the header first
        if not file_exists:
            header = ["Elapsed Seconds", "Layer Architecture", "Beta", "Learning Rate", "Error Epsilon", "Accuracy"]
            csvwriter.writerow(header)

        row = [elap_time, hyperparams["layer_sizes"], hyperparams["beta"], hyperparams["learning_rate"],
               hyperparams["error_limit"], accu]
        csvwriter.writerow(row)
