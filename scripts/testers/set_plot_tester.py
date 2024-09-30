import pandas as pd
import matplotlib.pyplot as plt


# Function to read the CSV and plot the data
def plot_csv_data(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Extract columns
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    z = data.iloc[:, 2]
    hue = data.iloc[:, 3]  # Assuming this is a value from 0 to 1 for color

    # Normalize hue values to be between 0 and 1 if necessary
    hue = (hue - hue.min()) / (hue.max() - hue.min())

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with color mapping
    scatter = ax.scatter(x, y, z, c=hue, cmap='hsv', marker='o')

    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Hue Value')

    # Set labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Scatter Plot from CSV Data')

    # Show plot
    plt.show()


# Example usage
if __name__ == "__main__":
    plot_csv_data('exercise2/data/set.csv')
