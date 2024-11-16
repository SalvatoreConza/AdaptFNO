import json
import numpy as np
import matplotlib.pyplot as plt

def plot_storm_trajectory(json_file):
    # Load data from JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract ground truth and prediction trajectories
    groundtruth_trajectory = []
    predicted_trajectory = []
    timestamps = []
    
    for timestamp in data:
        groundtruth_trajectory.append(data[timestamp]["groundtruth_hw"])
        predicted_trajectory.append(data[timestamp]["prediction_hw"])
        timestamps.append(timestamp)

    # Convert lists to numpy arrays for easier plotting
    groundtruth_trajectory = np.array(groundtruth_trajectory)
    predicted_trajectory = np.array(predicted_trajectory)
    
    # Plot the trajectories
    plt.figure(figsize=(10, 6))
    plt.plot(groundtruth_trajectory[:, 1], groundtruth_trajectory[:, 0], 'ro--', label='Ground Truth')
    plt.plot(predicted_trajectory[:, 1], predicted_trajectory[:, 0], 'bo--', label='Prediction')
    
    # Set axis limits to 160x160 grid
    plt.xlim(70, 130)
    plt.ylim(110, 130)
    plt.xticks(np.arange(70, 131, 10), fontsize=11)
    plt.yticks(np.arange(110, 131, 5), fontsize=11)

    # Add timestamps to the plot
    for i, timestamp in enumerate(timestamps):
        # Mark points with numbers
        plt.text(groundtruth_trajectory[i, 1], groundtruth_trajectory[i, 0] + 0.5, str(i), 
                 fontsize=10, color='red', ha='center', va='center')
        plt.text(predicted_trajectory[i, 1], predicted_trajectory[i, 0] + 0.5, str(i), 
                 fontsize=10, color='blue', ha='center', va='center')
    
    legend_text = "\n".join([f"{i}: {timestamp}" for i, timestamp in enumerate(timestamps)])
    plt.gcf().text(
        0.815, 0.5, legend_text, fontsize=10, va='center', ha='left', 
        color='black', bbox=dict(facecolor='white'),
        linespacing=1.5
    )
    plt.legend(frameon=True, fontsize=11, loc='upper left', framealpha=1, edgecolor='black')
    plt.xlabel("Longitude", fontsize=15)
    plt.ylabel("Latitude", fontsize=15)
    plt.title("Storm Trajectory - Ground Truth vs Prediction", fontsize=15)
    plt.tight_layout()
    
    # Save the plot as a PNG file
    plt.savefig('trajectory.png')
    plt.show()

# Usage example:
plot_storm_trajectory("yagi_trajectory.json")