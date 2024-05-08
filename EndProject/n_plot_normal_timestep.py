import numpy as np
import matplotlib.pyplot as plt

# Define number of datasets
n = 25

# Initialize arrays to store aggregated actual and predicted data
aggregated_actual_data = np.zeros((n, 3, 2000))
aggregated_predicted_data = np.zeros((n, 3, 2000))

predicted_data2 = np.genfromtxt('predicted_data_3.csv', delimiter=',', skip_header=1)
predicted_data2 = predicted_data2.T

actual_data2 = np.genfromtxt('actual_data_3.csv', delimiter=',', skip_header=1)
actual_data2 = actual_data2.T

# Load actual and predicted data for each dataset and compute the average
for i in range(n):
    # Load actual and predicted data
    actual_data = np.genfromtxt(f'actual_data_{i}.csv', delimiter=',', skip_header=1)
    predicted_data = np.genfromtxt(f'predicted_data_{i}.csv', delimiter=',', skip_header=1)
    
    # Compute the length of the data
    timesteps = np.arange(len(actual_data))
    
    # Aggregate data
    aggregated_actual_data[i, :, :len(actual_data)] = actual_data.T
    aggregated_predicted_data[i, :, :len(predicted_data)] = predicted_data.T

# Compute the average for each dataset
average_actual_data = np.mean(aggregated_actual_data, axis=0)
average_predicted_data = np.mean(aggregated_predicted_data, axis=0)

# Time steps
timesteps = np.arange(len(average_actual_data[0]))

# Create subplots
plt.figure(figsize=(12, 8))

# Plot X dimension
plt.subplot(3, 1, 1)
plt.plot(timesteps, actual_data2[0], label='Average Actual X', color='b', lw=0.8)
plt.plot(timesteps, average_predicted_data[0], '--', label='Average Predicted X', color='orange', lw=0.8)
plt.ylabel('X Value')
plt.title('Comparison of Average Actual and Predicted Data against Timestep')
plt.legend()
plt.grid(True)

# Plot Y dimension
plt.subplot(3, 1, 2)
plt.plot(timesteps, actual_data2[1], label='Average Actual Y', color='g', lw=0.8)
plt.plot(timesteps, average_predicted_data[1], '--', label='Average Predicted Y', color='orange', lw=0.8)
plt.ylabel('Y Value')
plt.legend()
plt.grid(True)

# Plot Z dimension
plt.subplot(3, 1, 3)
plt.plot(timesteps, actual_data2[2], label='Average Actual Z', color='r', lw=0.8)
plt.plot(timesteps, average_predicted_data[2], '--', label='Average Predicted Z', color='orange', lw=0.8)
plt.xlabel('Timestep')
plt.ylabel('Z Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()