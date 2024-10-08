import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt


# Function to extract and compute mean and std for Testing Accuracy
def process_testing_accuracy(df, group_by_column):
    grouped = df.groupby(group_by_column)
    results = {}

    for name, group in grouped:
        noise_levels = sorted(ast.literal_eval(group['Testing Accuracy'].iloc[0]).keys())
        mean_acc = []
        std_acc = []

        for noise in noise_levels:
            accuracies = [ast.literal_eval(acc)[noise] for acc in group['Testing Accuracy']]
            mean_acc.append(np.mean(accuracies))
            std_acc.append(np.std(accuracies))

        results[name] = (noise_levels, mean_acc, std_acc)

    return results


# Function to plot mean accuracy with standard deviation as shaded region
def plot_with_error_bars(results, parameter_name):
    plt.figure(figsize=(10, 6))
    for name, (noise_levels, mean_acc, std_acc) in results.items():
        noise_levels = [float(n) for n in noise_levels]  # Convert string noise levels to floats
        mean_acc = np.array(mean_acc)
        std_acc = np.array(std_acc)

        plt.plot(noise_levels, mean_acc, label=f'{parameter_name} = {name}')
        plt.fill_between(noise_levels, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2)

    plt.title(f'Testing Accuracy vs Noise Level ({parameter_name})')
    plt.xlabel('Noise Level')
    plt.ylabel('Testing Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("testing_accuracy_by_momentum_lineplot.png", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    # Load CSV data
    df = pd.read_csv('exercise3/outputs/classification_digits_themselves_momentum.csv')

    # Process and plot for each parameter
    for parameter in ['Momentum']: # , '', '', '']:
        results = process_testing_accuracy(df, parameter)
        plot_with_error_bars(results, parameter)
