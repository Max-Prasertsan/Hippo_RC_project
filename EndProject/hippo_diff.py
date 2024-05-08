import numpy as np
import matplotlib.pyplot as plt

# Load actual and predicted data
actual_data = np.genfromtxt('actual_data_hippo.csv', delimiter=',', skip_header=1)
predicted_data = np.genfromtxt('predicted_data_hippo.csv', delimiter=',', skip_header=1)

# Time steps
timesteps = np.arange(len(actual_data))

# Calculate absolute differences between actual and predicted data
diff = np.abs(actual_data - predicted_data)

# Set a threshold for differentiation
differentiation_threshold = 0.25  # You can adjust this threshold as needed

# Find where the differences are within the threshold
matching_indices = np.where(np.all(diff <= differentiation_threshold, axis=1))

# Extract matching data points
matching_timesteps = timesteps[matching_indices]
matching_actual_data = actual_data[matching_indices]
matching_predicted_data = predicted_data[matching_indices]

# Create subplots
plt.figure(figsize=(12, 8))

# Plot X dimension
plt.subplot(3, 1, 1)
plt.plot(matching_timesteps, matching_actual_data[:, 0], 'o', label='Actual X', color='b')
plt.plot(matching_timesteps, matching_predicted_data[:, 0], 'o', label='Predicted X', color='orange')
plt.ylabel('X Value')
plt.title('Comparison of Matching Data Points within Threshold')
plt.legend()
plt.grid(True)

# Plot Y dimension
plt.subplot(3, 1, 2)
plt.plot(matching_timesteps, matching_actual_data[:, 1], 'o', label='Actual Y', color='g')
plt.plot(matching_timesteps, matching_predicted_data[:, 1], 'o', label='Predicted Y', color='orange')
plt.ylabel('Y Value')
plt.legend()
plt.grid(True)

# Plot Z dimension
plt.subplot(3, 1, 3)
plt.plot(matching_timesteps, matching_actual_data[:, 2], 'o', label='Actual Z', color='r')
plt.plot(matching_timesteps, matching_predicted_data[:, 2], 'o', label='Predicted Z', color='orange')
plt.xlabel('Timestep')
plt.ylabel('Z Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

