import numpy as np
import matplotlib.pyplot as plt

class LossFunction:
    @staticmethod
    def absolute_difference(actual, predicted):
        return np.abs(actual - predicted)

# Define number of datasets
n = 50

# Initialize array to store absolute differences for each dataset
absolute_diff_all_datasets = []

# Load actual and predicted data for each dataset and compute the absolute difference
for i in range(n):
    actual_data = np.genfromtxt(f'actual_data_V1_{i}.csv', delimiter=',', skip_header=1)
    predicted_data = np.genfromtxt(f'predicted_data_V1_{i}.csv', delimiter=',', skip_header=1)
    
    # Calculate absolute difference
    absolute_diff = LossFunction.absolute_difference(actual_data, predicted_data)
    absolute_diff_all_datasets.append(absolute_diff)

# Convert list to numpy array for easy manipulation
absolute_diff_all_datasets = np.array(absolute_diff_all_datasets)

# Calculate average absolute difference over datasets
average_absolute_diff = np.mean(absolute_diff_all_datasets, axis=0)

# Plot the average absolute difference for each dimension separately
fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

for i, ax in enumerate(axs):
    ax.plot(average_absolute_diff[:, i], label=f'Dimension {"XYZ"[i]}', color = 'r')
    ax.set_ylabel('Absolute Difference')
    ax.set_title(f'Average Absolute Difference for Dimension {"XYZ"[i]}')  # Change title
    ax.grid(True)
    ax.legend()

axs[-1].set_xlabel('Time Step')

plt.tight_layout()
plt.show()
