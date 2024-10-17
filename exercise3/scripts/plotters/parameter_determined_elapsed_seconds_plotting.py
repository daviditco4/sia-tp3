import sys

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python parameter_determined_elapsed_seconds_plotting.py <data_csv_file> <varying_hyperparameter_name>")
        sys.exit(1)

    # Read the CSV file
    data = pd.read_csv(sys.argv[1])
    varying_hyperparam = sys.argv[2]

    # Group the data by the varying hyperparameter
    grouped = data.groupby(varying_hyperparam)

    # Calculate the mean and standard deviation for the 'Elapsed Seconds' for each group
    means = grouped['Elapsed Seconds'].mean()
    stds = grouped['Elapsed Seconds'].std()

    # Create a bar plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(means.index, means, yerr=stds, capsize=5, alpha=0.75, color='lightblue')

    # Add labels and title
    plt.xlabel(varying_hyperparam)
    plt.ylabel('Elapsed Seconds')
    plt.title(f'Average Elapsed Seconds vs {varying_hyperparam}')

    # Adjust x-axis labels to avoid overlap by rotating and setting padding
    plt.xticks(rotation=45, ha="right")

    # Show values (means) on top of the bars with a slight horizontal shift to avoid overlap with error bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2 + 0.2, yval, round(yval, 2), ha='center', va='bottom')

    # Ensure the layout is tight so labels and plot elements don't overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        f"digits_themselves_{varying_hyperparam.lower().replace(" ", "_")}_determined_elapsed_seconds_plot.png",
        dpi=300, bbox_inches='tight')
    plt.close()
