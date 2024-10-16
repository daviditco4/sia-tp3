import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python training_testing_accuracy_vs_epoch_plotting.py <data_csv_file>")
        sys.exit(1)

    # Read the CSV file
    data = pd.read_csv(sys.argv[1])

    # Extract training and testing accuracies
    training_accuracies = data['Training Accuracy'].apply(eval).values  # convert strings of lists to actual lists
    testing_accuracies = data['Testing Accuracy'].apply(eval).values

    # Store all epoch values for plotting
    all_epochs = []
    all_training = []
    all_testing = []

    # For each row, extract the accuracies and their corresponding epoch count
    for train_acc, test_acc in zip(training_accuracies, testing_accuracies):
        epochs = np.arange(1, len(train_acc) + 1)  # X-axis values as epoch counts
        all_epochs.append(epochs)
        all_training.append(train_acc)
        all_testing.append(test_acc)

    # Find the maximum number of epochs across all rows to handle varying lengths
    max_epochs = max([len(epochs) for epochs in all_epochs])

    # Create empty lists to accumulate mean and std values for each epoch
    train_means = []
    test_means = []
    train_stds = []
    test_stds = []

    # Calculate means and standard deviations for each epoch index
    for epoch in range(1, max_epochs + 1):
        train_epoch_values = []
        test_epoch_values = []

        # Collect all training and testing accuracy values for the current epoch
        for i, epochs in enumerate(all_epochs):
            if epoch <= len(epochs):  # Only include epochs within the current row's range
                train_epoch_values.append(all_training[i][epoch - 1])
                test_epoch_values.append(all_testing[i][epoch - 1])

        # Calculate mean and std for the current epoch
        train_means.append(np.mean(train_epoch_values))
        test_means.append(np.mean(test_epoch_values))
        train_stds.append(np.std(train_epoch_values))
        test_stds.append(np.std(test_epoch_values))

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot Training Accuracy
    plt.plot(np.arange(1, max_epochs + 1), train_means, label='Training Accuracy', color='blue')
    plt.fill_between(np.arange(1, max_epochs + 1),
                     np.array(train_means) - np.array(train_stds),
                     np.array(train_means) + np.array(train_stds),
                     color='blue', alpha=0.2)

    # Plot Testing Accuracy
    plt.plot(np.arange(1, max_epochs + 1), test_means, label='Testing Accuracy', color='red')
    plt.fill_between(np.arange(1, max_epochs + 1),
                     np.array(test_means) - np.array(test_stds),
                     np.array(test_means) + np.array(test_stds),
                     color='red', alpha=0.2)

    # Add plot labels and legend
    plt.title('Training and Testing Accuracy with Confidence Bands')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig("training_testing_accuracy_vs_epoch_plot4.png", dpi=300, bbox_inches='tight')
    plt.close()
