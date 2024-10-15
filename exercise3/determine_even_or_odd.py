import csv
import os
import sys
import time

import numpy as np

from utils import read_digits_from_txt, read_hyperparameters_from_json, train_perceptron


def append_results_to_csv(file_path, elap_time, hyperparams, iters, train_accu):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # If the file is new, write the header first
        if not file_exists:
            header = ["Elapsed Seconds", "Layer Architecture", "Beta", "Learning Rate", "Momentum", "Error Epsilon",
                      "Iterations", "Training Accuracy"]
            csvwriter.writerow(header)

        row = [elap_time, hyperparams["layer_sizes"], hyperparams["beta"], hyperparams["learning_rate"],
               hyperparams["momentum"] if 'momentum' in hyperparams else 0, hyperparams["error_limit"], iters, train_accu]
        csvwriter.writerow(row)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python determine_even_or_odd.py <digits_txt_file> <hyperparameters_json_file> <output_csv_file>")
        sys.exit(1)

    digits_txt_file = sys.argv[1]
    hyperparameters_json_file = sys.argv[2]
    output_csv_file = sys.argv[3]

    # Read the digits and labels (even/odd classification)
    digits = read_digits_from_txt(digits_txt_file)
    labels = np.array([i % 2 for i in range(10)]).reshape(-1, 1)  # Even/odd labels

    # Read the hyperparameters from JSON
    hyperparameters = read_hyperparameters_from_json(hyperparameters_json_file)

    # Start timing
    start_time = time.time()

    # Train the perceptron
    mlp, iterations = train_perceptron(digits, labels, hyperparameters)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Predict on the training set to calculate accuracy
    predictions = mlp.predict(digits)
    predicted_labels = np.round(predictions)
    training_accuracy = np.mean(predicted_labels == labels)

    # Append results to CSV
    append_results_to_csv(output_csv_file, elapsed_time, hyperparameters, iterations, training_accuracy)

    print(f"Training completed in {elapsed_time:.2f} seconds with {training_accuracy * 100:.2f}% accuracy")
