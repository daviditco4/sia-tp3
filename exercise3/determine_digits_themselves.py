import sys
import time

import numpy as np

from utils import read_digits_from_txt, read_hyperparameters_from_json, train_perceptron, \
    append_results_to_csv


def generate_labels_for_digits(num_digits=10):
    """ Create one-hot encoded labels for digits 0-9 """
    lbls = np.eye(num_digits)
    return lbls


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
