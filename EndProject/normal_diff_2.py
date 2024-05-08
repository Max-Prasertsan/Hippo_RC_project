import numpy as np
import matplotlib.pyplot as plt

# Load actual and predicted data
actual_data = np.genfromtxt('actual_data.csv', delimiter=',', skip_header=1)
predicted_data = np.genfromtxt('predicted_data.csv', delimiter=',', skip_header=1)

# Calculate absolute differences between actual and predicted data
diff = np.abs(actual_data - predicted_data)

# Set a threshold for differentiation
differentiation_threshold = 0.5  # You can adjust this threshold as needed

# Find where the differences are within the threshold
matching_indices = np.where(np.all(diff <= differentiation_threshold, axis=1))

# Extract matching data points
matching_diff = diff[matching_indices]

# Create subplots
plt.figure(figsize=(12, 6))

# Plot histogram for X dimension
plt.subplot(1, 3, 1)
plt.hist(matching_diff[:, 0], bins=20, color='b', alpha=0.7)
plt.xlabel('Absolute Difference (X)')
plt.ylabel('Frequency')
plt.title('Histogram of Matching Results (X)')
plt.grid(True)

# Plot histogram for Y dimension
plt.subplot(1, 3, 2)
plt.hist(matching_diff[:, 1], bins=20, color='g', alpha=0.7)
plt.xlabel('Absolute Difference (Y)')
plt.ylabel('Frequency')
plt.title('Histogram of Matching Results (Y)')
plt.grid(True)

# Plot histogram for Z dimension
plt.subplot(1, 3, 3)
plt.hist(matching_diff[:, 2], bins=20, color='r', alpha=0.7)
plt.xlabel('Absolute Difference (Z)')
plt.ylabel('Frequency')
plt.title('Histogram of Matching Results (Z)')
plt.grid(True)

plt.tight_layout()
plt.show()