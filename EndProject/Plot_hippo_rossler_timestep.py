import numpy as np
import matplotlib.pyplot as plt

# Load actual and predicted data
actual_data = np.genfromtxt('actual_data_rossler.csv', delimiter=',', skip_header=1)
predicted_data = np.genfromtxt('predicted_data_rossler.csv', delimiter=',', skip_header=1)

# Time steps
timesteps = np.arange(len(actual_data))

# Create subplots
plt.figure(figsize=(12, 8))

# Plot X dimension
plt.subplot(3, 1, 1)
plt.plot(timesteps, actual_data[:,0], label='Actual X', color='b', lw=0.8)
plt.plot(timesteps, predicted_data[:,0], '--', label='Predicted X', color='orange', lw=0.8)
plt.ylabel('X Value')
plt.title('Comparison of Actual and Predicted Data against Timestep')
plt.legend()
plt.grid(True)

# Plot Y dimension
plt.subplot(3, 1, 2)
plt.plot(timesteps, actual_data[:,1], label='Actual Y', color='g', lw=0.8)
plt.plot(timesteps, predicted_data[:,1], '--', label='Predicted Y', color='orange', lw=0.8)
plt.ylabel('Y Value')
plt.legend()
plt.grid(True)

# Plot Z dimension
plt.subplot(3, 1, 3)
plt.plot(timesteps, actual_data[:,2], label='Actual Z', color='r', lw=0.8)
plt.plot(timesteps, predicted_data[:,2], '--', label='Predicted Z', color='orange', lw=0.8)
plt.xlabel('Timestep')
plt.ylabel('Z Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()