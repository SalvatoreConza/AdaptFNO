import json
import numpy as np
import matplotlib.pyplot as plt

def plot_storm_trajectory_with_error(json_file):
    # Load data from JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract ground truth and prediction trajectories and timestamps
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
    
    # Calculate the Euclidean distance error at each time step
    errors = np.linalg.norm(groundtruth_trajectory - predicted_trajectory, axis=1) * 30 # each pixel is 30km
    
    # Create subplots: 2 rows, 1 column
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot the trajectories on the first subplot
    ax1.plot(groundtruth_trajectory[:, 1], groundtruth_trajectory[:, 0], 'r--', label='Ground Truth')
    ax1.plot(predicted_trajectory[:, 1], predicted_trajectory[:, 0], 'b--', label='Prediction')
    
    # Add arrows between each point in the ground truth trajectory
    for i in range(len(groundtruth_trajectory) - 1):
        ax1.annotate('', xy=(groundtruth_trajectory[i + 1, 1], groundtruth_trajectory[i + 1, 0]), 
                     xytext=(groundtruth_trajectory[i, 1], groundtruth_trajectory[i, 0]),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    # Add arrows between each point in the predicted trajectory
    for i in range(len(predicted_trajectory) - 1):
        ax1.annotate('', xy=(predicted_trajectory[i + 1, 1], predicted_trajectory[i + 1, 0]), 
                     xytext=(predicted_trajectory[i, 1], predicted_trajectory[i, 0]),
                     arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    # Draw compass on the top-right corner of the first subplot
    compass_x, compass_y = 112.5, 127.5
    ax1.annotate('', xy=(compass_x, compass_y + 1.5), xytext=(compass_x, compass_y),
                arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5))
    ax1.text(compass_x, compass_y + 1.5, 'N', ha='center', va='bottom', fontsize=10)
    # East arrow
    ax1.annotate('', xy=(compass_x + 1.5, compass_y), xytext=(compass_x, compass_y),
                arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5))
    ax1.text(compass_x + 1.5, compass_y, 'E', ha='left', va='center', fontsize=10)
    # South arrow
    ax1.annotate('', xy=(compass_x, compass_y - 1.5), xytext=(compass_x, compass_y),
                arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5))
    ax1.text(compass_x, compass_y - 1.5, 'S', ha='center', va='top', fontsize=10)
    # West arrow
    ax1.annotate('', xy=(compass_x - 1.5, compass_y), xytext=(compass_x, compass_y),
                arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5))
    ax1.text(compass_x - 1.5, compass_y, 'W', ha='right', va='center', fontsize=10)

    # Set axis limits and ticks
    ax1.set_xlim(70, 130)
    ax1.set_ylim(105, 130)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xticks(np.arange(70, 131, 5))
    ax1.tick_params(axis='x', labelsize=11)
    ax1.set_xlabel("Longitude", fontsize=15)
    ax1.set_yticks(np.arange(105, 131, 5))
    ax1.tick_params(axis='y', labelsize=11)
    ax1.set_ylabel("Latitude", fontsize=15)
    ax1.grid(True)
    ax1.legend(frameon=True, fontsize=11, loc='lower left', framealpha=1, edgecolor='black')
    ax1.set_title("Yagi's Trajectory - Ground Truth vs Prediction", fontsize=15)
    
    # Add timestamps as markers
    for i, timestamp in enumerate(timestamps):
        ax1.text(groundtruth_trajectory[i, 1], groundtruth_trajectory[i, 0] + 0.5, str(i), 
                 fontsize=10, color='red', ha='center', va='center')
        ax1.text(predicted_trajectory[i, 1], predicted_trajectory[i, 0] + 0.5, str(i), 
                 fontsize=10, color='blue', ha='center', va='center')
    
    # Add a legend for the timestamps on the right side
    legend_text = "\n".join([f"{i}: {timestamp}" for i, timestamp in enumerate(timestamps)])
    fig.text(
        0.815, 0.9, legend_text, fontsize=10, va='top', ha='left', 
        color='black', bbox=dict(facecolor='white'),
        linespacing=1.5
    )
    
    # Plot the error over time on the second subplot
    ax2.plot(range(len(errors)), errors, 'm--s', label='Error')
    ax2.set_xticks(range(len(timestamps)))
    ax2.tick_params(axis='x', labelsize=11)
    ax2.set_xticklabels(list(range(len(timestamps))))
    ax2.set_ylabel("Approx. Error (km)", fontsize=15)
    ax2.set_xlabel("Timestamp", fontsize=15)
    ax2.set_title("Trajectory Error Over Time", fontsize=15)
    ax2.legend()
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig('trajectory_with_error.png', bbox_inches='tight')
    plt.show()

# Usage example:
plot_storm_trajectory_with_error("yagi_trajectory.json")