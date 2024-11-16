import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
import json

# Load data from a JSON file
with open('evaluate_main.json', 'r') as file:
    data1 = json.load(file)

# Convert data to DataFrame
df1 = pd.DataFrame(data1)

# Plotting RMSE and MSE distributions by step in subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))

# Set the box width and colors
box_width = 3  # Adjust this value as needed

with open('evaluate_fno3d.json', 'r') as file:
    data2 = json.load(file)

df2 = pd.DataFrame(data2)

# Calculate median RMSE and MSE for FNO3D
median_rmse = [df2[df2['prediction_hours'] == hour]['rmse'].median() for hour in sorted(df2['prediction_hours'].unique())]
median_mse = [df2[df2['prediction_hours'] == hour]['mse'].median() for hour in sorted(df2['prediction_hours'].unique())]
hours = sorted(df1['prediction_hours'].unique())

# Define legend patches and lines
box_rmse_patch = mpatches.Patch(color='blue', label='FourXNet')
box_mse_patch = mpatches.Patch(color='green', label='FourXNet')
line_rmse_patch = mlines.Line2D([], [], color='red', linestyle='--', marker='s', label='FNO3D')
line_mse_patch = mlines.Line2D([], [], color='orange', linestyle='--', marker='s', label='FNO3D')

# RMSE Boxplot
ax1.boxplot(
    [df1[df1['prediction_hours'] == hour]['rmse'] for hour in sorted(df1['prediction_hours'].unique())],
    positions=sorted(df1['prediction_hours'].unique()),
    widths=box_width,
    patch_artist=True,
    boxprops=dict(color='blue', facecolor='blue', edgecolor="black"),
    medianprops=dict(color='black')
)
ax1.plot(hours, median_rmse, color='red', linestyle='--', marker='s', label='FNO3D RMSE')
ax1.legend(handles=[box_rmse_patch, line_rmse_patch], loc='upper left')
ax1.set_xlabel('Prediction Hours', fontsize=12)
ax1.set_ylabel('RMSE', fontsize=12)

# MSE Boxplot
ax2.boxplot(
    [df1[df1['prediction_hours'] == hour]['mse'] for hour in sorted(df1['prediction_hours'].unique())],
    positions=sorted(df1['prediction_hours'].unique()),
    widths=box_width,
    patch_artist=True,
    boxprops=dict(color='green', facecolor='green', edgecolor="black"),
    medianprops=dict(color='black')
)
ax2.plot(hours, median_mse, color='orange', linestyle='--', marker='s', label='FNO3D MSE')
ax2.legend(handles=[box_mse_patch, line_mse_patch], loc='upper left')
ax2.set_xlabel('Prediction Hours', fontsize=12)
ax2.set_ylabel('MSE', fontsize=12)

plt.tight_layout()
plt.savefig('dist.png')