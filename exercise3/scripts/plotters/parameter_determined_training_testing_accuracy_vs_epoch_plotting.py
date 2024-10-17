import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python parameter_determined_training_testing_accuracy_vs_epoch_plotting.py <data_csv_file> <varying_hyperparameter_name>")
        sys.exit(1)

    # Read the CSV file
    data = pd.read_csv(sys.argv[1])
    varying_hyperparam = sys.argv[2]

    # Convert 'Training Accuracy' and 'Testing Accuracy' columns from strings to lists of floats
    data['Training Accuracy'] = data['Training Accuracy'].apply(eval)
    data['Testing Accuracy'] = data['Testing Accuracy'].apply(eval)

    # Group data by the chosen hyperparameter's column
    unique_values = data[varying_hyperparam].unique()

    # Prepare the plot
    plt.figure(figsize=(12, 8))

    # For each unique value of the chosen hyperparameter, compute and plot the mean and std for Training and Testing accuracy
    for val in unique_values:
        subset = data[data[varying_hyperparam] == val]

        # Determine the maximum number of epochs based on the longest list in the subset
        max_epochs = max(subset['Training Accuracy'].apply(len))

        # Initialize lists to store all accuracy values for Training and Testing
        all_training_accuracies = []
        all_testing_accuracies = []

        # Collect all accuracies and pad shorter lists with NaN to handle lists of different lengths
        for train_acc, test_acc in zip(subset['Training Accuracy'], subset['Testing Accuracy']):
            padded_train_acc = train_acc + [np.nan] * (max_epochs - len(train_acc))
            padded_test_acc = test_acc + [np.nan] * (max_epochs - len(test_acc))

            all_training_accuracies.append(padded_train_acc)
            all_testing_accuracies.append(padded_test_acc)

        all_training_accuracies = np.array(all_training_accuracies)
        all_testing_accuracies = np.array(all_testing_accuracies)

        # Compute mean and std for training and testing accuracy, ignoring NaN values
        train_acc_means = np.nanmean(all_training_accuracies, axis=0)
        train_acc_stds = np.nanstd(all_training_accuracies, axis=0)

        test_acc_means = np.nanmean(all_testing_accuracies, axis=0)
        test_acc_stds = np.nanstd(all_testing_accuracies, axis=0)

        # Generate epoch indices (1-based)
        epochs = np.arange(1, max_epochs + 1)

        # Plot Training Accuracy mean and std shadow for this hyperparameter value
        plt.plot(epochs, train_acc_means, label=f'Training - {varying_hyperparam}: {val}')
        plt.fill_between(epochs, train_acc_means - train_acc_stds, train_acc_means + train_acc_stds, alpha=0.2)

        # # Plot Testing Accuracy mean and std shadow for this hyperparameter value
        # plt.plot(epochs, test_acc_means, label=f'Testing - {varying_hyperparam}: {val}')
        # plt.fill_between(epochs, test_acc_means - test_acc_stds, test_acc_means + test_acc_stds, alpha=0.2)

    # Limit X-axis to 2500 epochs
    plt.xlim([1, 2500])

    # Add labels, title, and legend
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Training Accuracy for varying {varying_hyperparam}') # (f'Training and Testing Accuracy for varying {varying_hyperparam}')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(
        f"digits_themselves_{varying_hyperparam.lower().replace(" ", "_")}_determined_training_accuracy_vs_epoch_plot.png",
        dpi=300, bbox_inches='tight')
    plt.close()
