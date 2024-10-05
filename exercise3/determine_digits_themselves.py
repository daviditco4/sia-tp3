import csv
import json
import numpy as np
import os
import random
import sys
import time

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


def generate_labels_for_digits(num_digits=10):
    """ Create one-hot encoded labels for digits 0-9 """
    lbls = np.eye(num_digits)
    return lbls


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


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python determine_digits_themselves.py <digits_txt_file> <hyperparameters_json_file> <output_csv_file>")
        sys.exit(1)

    digits_txt_file = sys.argv[1]
    hyperparameters_json_file = sys.argv[2]
    output_csv_file = sys.argv[3]

    # Start timing
    start_time = time.time()

    # Read the digits from the TXT file
    digits = read_digits_from_txt(digits_txt_file)

    # Read hyperparameters from JSON
    hyperparameters = read_hyperparameters_from_json(hyperparameters_json_file)

    # Generate one-hot encoded labels for digits 0 to 9
    labels = generate_labels_for_digits()

    # Train the perceptron
    mlp = train_perceptron(digits, labels, hyperparameters)

    # Predict on the training set to calculate accuracy
    predictions = mlp.predict(digits)
    predicted_labels = np.argmax(predictions, axis=1)  # Convert outputs back to digit labels
    true_labels = np.argmax(labels, axis=1)

    accuracy = np.mean(predicted_labels == true_labels)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Append results to CSV
    append_results_to_csv(output_csv_file, elapsed_time, hyperparameters, accuracy)

    print(f"Training completed in {elapsed_time:.2f} seconds with {accuracy * 100:.2f}% accuracy.")
