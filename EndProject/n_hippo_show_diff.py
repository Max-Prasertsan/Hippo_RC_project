import numpy as np
import matplotlib.pyplot as plt

# Define number of datasets
n = 50

# Initialize arrays to store actual and predicted data for all datasets
all_actual_data = []
all_predicted_data = []

# Load actual and predicted data for each dataset
for i in range(n):
    actual_data_i = np.genfromtxt(f'actual_data_hippo_V1_{i}.csv', delimiter=',', skip_header=1)
    predicted_data_i = np.genfromtxt(f'predicted_data_hippo_V1_{i}.csv', delimiter=',', skip_header=1)
    all_actual_data.append(actual_data_i)
    all_predicted_data.append(predicted_data_i)

# Convert lists to numpy arrays for easy manipulation
all_actual_data = np.array(all_actual_data)
all_predicted_data = np.array(all_predicted_data)

# Compute the average actual and predicted data
average_actual_data = np.mean(all_actual_data, axis=0)
average_predicted_data = np.mean(all_predicted_data, axis=0)

# Time steps
timesteps = np.arange(len(average_actual_data))

# Calculate absolute differences between average actual and predicted data
diff = np.abs(average_actual_data - average_predicted_data)

# Set a threshold for differentiation
differentiation_threshold = 0.6  # You can adjust this threshold as needed

# Find where the differences exceed the threshold
differentiated_indices = np.argwhere(diff > differentiation_threshold)

# Flatten the array of indices
differentiated_indices = differentiated_indices.flatten()

# Create subplots
plt.figure(figsize=(12, 8))

# Plot X dimension
plt.subplot(3, 1, 1)
plt.plot(timesteps, average_actual_data[:, 0], label='Average Actual X', color='b', lw=0.8)
plt.plot(timesteps, average_predicted_data[:, 0], '--', label='Average Predicted X', color='orange', lw=0.8)
plt.ylabel('X Value')
plt.title('Comparison of Average Actual and Predicted Data against Timestep')
plt.legend(loc='upper right')
plt.grid(True)
for idx in differentiated_indices:
    plt.axvline(x=idx, color='r', linestyle='--', alpha=0.5)

# Plot Y dimension
plt.subplot(3, 1, 2)
plt.plot(timesteps, average_actual_data[:, 1], label='Average Actual Y', color='g', lw=0.8)
plt.plot(timesteps, average_predicted_data[:, 1], '--', label='Average Predicted Y', color='orange', lw=0.8)
plt.ylabel('Y Value')
plt.legend(loc='upper right')
plt.grid(True)
for idx in differentiated_indices:
    plt.axvline(x=idx, color='r', linestyle='--', alpha=0.5)

# Plot Z dimension
plt.subplot(3, 1, 3)
plt.plot(timesteps, average_actual_data[:, 2], label='Average Actual Z', color='r', lw=0.8)
plt.plot(timesteps, average_predicted_data[:, 2], '--', label='Average Predicted Z', color='orange', lw=0.8)
plt.xlabel('Timestep')
plt.ylabel('Z Value')
plt.legend(loc='upper right')
plt.grid(True)
for idx in differentiated_indices:
    plt.axvline(x=idx, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
