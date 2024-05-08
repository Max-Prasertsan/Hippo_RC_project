import numpy as np
import matplotlib.pyplot as plt

class LossFunction:
    @staticmethod
    def absolute_difference(actual, predicted):
        return np.abs(actual - predicted)

# Define number of datasets
n = 50

# Initialize arrays to store absolute differences for each dataset (hippo and normal)
absolute_diff_all_datasets_hippo = []
absolute_diff_all_datasets_normal = []

# Load actual and predicted data for each dataset and compute the absolute difference
for i in range(n):
    actual_data_hippo = np.genfromtxt(f'actual_data_hippo_V14_{i}.csv', delimiter=',', skip_header=1)
    predicted_data_hippo = np.genfromtxt(f'predicted_data_hippo_V14_{i}.csv', delimiter=',', skip_header=1)
    actual_data_normal = np.genfromtxt(f'actual_data_V14_{i}.csv', delimiter=',', skip_header=1)
    predicted_data_normal = np.genfromtxt(f'predicted_data_V14_{i}.csv', delimiter=',', skip_header=1)
    
    # Calculate absolute difference for hippo datasets
    absolute_diff_hippo = LossFunction.absolute_difference(actual_data_hippo, predicted_data_hippo)
    absolute_diff_all_datasets_hippo.append(absolute_diff_hippo)
    
    # Calculate absolute difference for normal datasets
    absolute_diff_normal = LossFunction.absolute_difference(actual_data_normal, predicted_data_normal)
    absolute_diff_all_datasets_normal.append(absolute_diff_normal)

# Convert lists to numpy arrays for easy manipulation
absolute_diff_all_datasets_hippo = np.array(absolute_diff_all_datasets_hippo)
absolute_diff_all_datasets_normal = np.array(absolute_diff_all_datasets_normal)

# Calculate average absolute difference over datasets for hippo and normal
average_absolute_diff_hippo = np.mean(absolute_diff_all_datasets_hippo, axis=0)
average_absolute_diff_normal = np.mean(absolute_diff_all_datasets_normal, axis=0)

# Plot the average absolute difference for each dimension separately
fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Plot average absolute difference for normal datasets
axs[0].plot(average_absolute_diff_normal[:, 0], label='Dimension X (Normal)', color='red')
axs[1].plot(average_absolute_diff_normal[:, 1], label='Dimension Y (Normal)', color='red')
axs[2].plot(average_absolute_diff_normal[:, 2], label='Dimension Z (Normal)', color='red')

# Plot average absolute difference for hippo datasets
axs[0].plot(average_absolute_diff_hippo[:, 0], label='Dimension X (Hippo)', color='blue')
axs[1].plot(average_absolute_diff_hippo[:, 1], label='Dimension Y (Hippo)', color='blue')
axs[2].plot(average_absolute_diff_hippo[:, 2], label='Dimension Z (Hippo)', color='blue')


# Set titles and labels
for i, ax in enumerate(axs):
    ax.set_ylabel('Absolute Difference')
    ax.grid(True)
    ax.legend()

axs[-1].set_xlabel('Time Step')

plt.tight_layout()
plt.show()

