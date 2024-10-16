import csv
import os
import random
import sys
import time

import numpy as np

from utils import read_digits_from_txt, read_hyperparameters_from_json, train_perceptron


def generate_labels_for_digits(num_digits=10):
    """ Create one-hot encoded labels for digits 0-9 """
    return np.eye(num_digits)


def apply_noise_to_bitmaps(bitmaps, noise_level):
    """
    Applies noise to the bitmaps by flipping a fraction of the bits.
    noise_level: The fraction of bits to flip (between 0 and 1).
    """
    noisy_bitmaps = bitmaps.copy()
    num_bits = noisy_bitmaps.shape[1]  # Number of bits in each bitmap (should be 35 for 5x7)
    num_digits = noisy_bitmaps.shape[0]  # Number of digits (typically 10)

    # Calculate how many bits to flip based on noise level
    num_bits_to_flip = int(num_bits * noise_level)

    for digit_index in range(num_digits):
        # Randomly select which bits to flip
        flip_indices = random.sample(range(num_bits), num_bits_to_flip)
        for bit_index in flip_indices:
            # Flip the bit (0 -> 1, 1 -> 0)
            noisy_bitmaps[digit_index, bit_index] = 1 - noisy_bitmaps[digit_index, bit_index]

    return noisy_bitmaps


def test_perceptron(p, dgts):
    accuracies = {}

    for i in range(10):
        noise_level = (i + 1) * 0.05
        noisy_digits = apply_noise_to_bitmaps(dgts, noise_level)
        preds = p.predict(noisy_digits)
        pred_labels = np.argmax(preds, axis=1)
        lbls = np.argmax(generate_labels_for_digits(), axis=1)
        accu = np.mean(pred_labels == lbls)
        accuracies[f"{noise_level:.2f}"] = accu

    return accuracies


def append_results_to_csv(file_path, elap_time, hyperparams, iters, train_accu, test_accu):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # If the file is new, write the header first
        if not file_exists:
            header = ["Elapsed Seconds", "Layer Architecture", "Beta", "Learning Rate", "Momentum", "Error Epsilon",
                      "Iterations", "Training Accuracy", "Testing Accuracy"]
            csvwriter.writerow(header)

        row = [elap_time, hyperparams["layer_sizes"], hyperparams["beta"], hyperparams["learning_rate"],
               hyperparams["momentum"] if 'momentum' in hyperparams else 0, hyperparams["error_limit"], iters,
               train_accu, test_accu]
        csvwriter.writerow(row)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python determine_digits_themselves.py <digits_txt_file> <hyperparameters_json_file> <output_csv_file>")
        sys.exit(1)

    digits_txt_file = sys.argv[1]
    hyperparameters_json_file = sys.argv[2]
    output_csv_file = sys.argv[3]

    # Read the digits from the TXT file
    digits = read_digits_from_txt(digits_txt_file)

    # Read hyperparameters from JSON
    hyperparameters = read_hyperparameters_from_json(hyperparameters_json_file)

    # Generate one-hot encoded labels for digits 0 to 9
    labels = generate_labels_for_digits()

    # Start timing
    start_time = time.time()

    # Train the perceptron
    mlp, iterations = train_perceptron(digits, labels, hyperparameters)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Predict on the training set to calculate accuracy
    predictions = mlp.predict(digits)
    predicted_labels = np.argmax(predictions, axis=1)  # Convert outputs back to digit labels
    true_labels = np.argmax(labels, axis=1)

    training_accuracy = np.mean(predicted_labels == true_labels)
    testing_accuracy = test_perceptron(mlp, digits)

    # Append results to CSV
    append_results_to_csv(output_csv_file, elapsed_time, hyperparameters, iterations, training_accuracy, testing_accuracy)

    print(f"Training completed in {elapsed_time:.2f} seconds with {training_accuracy * 100:.2f}% training accuracy")
