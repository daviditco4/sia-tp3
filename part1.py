import numpy as np
import json
import csv
import time
import sys
from MultilayerPerceptron import Perceptron


def read_digits_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the 5x7 bitmaps for each digit (assuming digits are in order 0 to 9)
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


def is_even(digit):
    return 0 if digit % 2 == 0 else 1


def train_perceptron(x, y, hyperparams):
    # Initialize perceptron using the hyperparameters
    p = Perceptron(layer_sizes=hyperparams["layer_sizes"], beta=hyperparams["beta"],
                   learning_rate=hyperparams["learning_rate"])

    # Train the perceptron
    best_weights, min_error = p.train(x, y, max_iterations=np.inf, error_limit=hyperparams["error_limit"])

    print(min_error)
    return p


def append_results_to_csv(file_path, hyperparams, accu, elap_time):
    with open(file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        row = [
            elap_time,
            hyperparams["layer_sizes"],
            hyperparams["beta"],
            hyperparams["error_limit"],
            hyperparams["learning_rate"],
            accu
        ]
        csvwriter.writerow(row)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python classify_digits.py <digits_txt_file> <hyperparameters_json_file> <output_csv_file>")
        sys.exit(1)

    digits_txt_file = sys.argv[1]
    hyperparameters_json_file = sys.argv[2]
    output_csv_file = sys.argv[3]

    # Start timing
    start_time = time.time()

    # Read the digits and labels (even/odd classification)
    digits = read_digits_from_txt(digits_txt_file)
    labels = np.array([i % 2 for i in range(10)]).reshape(-1, 1)  # Even/odd labels

    # Read the hyperparameters from JSON
    hyperparameters = read_hyperparameters_from_json(hyperparameters_json_file)

    # Train the perceptron
    mlp = train_perceptron(digits, labels, hyperparameters)

    # Predict on the training set to calculate accuracy
    predictions = mlp.predict(digits)
    predicted_labels = np.round(predictions)
    accuracy = np.mean(predicted_labels == labels) * 100

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Append results to CSV
    append_results_to_csv(output_csv_file, hyperparameters, accuracy, elapsed_time)

    print(f"Training completed in {elapsed_time:.2f} seconds with {accuracy:.2f}% accuracy.")
