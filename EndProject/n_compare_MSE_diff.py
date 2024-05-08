import numpy as np
import matplotlib.pyplot as plt

class LossFunction:
    @staticmethod
    def mean_squared_error(actual, predicted):
        return np.mean((actual - predicted) ** 2, axis=1)

# Define number of datasets
n = 50

# Initialize arrays to store MSE for each dataset (hippo and normal)
mse_all_datasets_hippo = []
mse_all_datasets_normal = []

# Load actual and predicted data for each dataset and compute the MSE
for i in range(n):
    actual_data_hippo = np.genfromtxt(f'actual_data_hippo_V14_{i}.csv', delimiter=',', skip_header=1)
    predicted_data_hippo = np.genfromtxt(f'predicted_data_hippo_V14_{i}.csv', delimiter=',', skip_header=1)
    actual_data_normal = np.genfromtxt(f'actual_data_V14_{i}.csv', delimiter=',', skip_header=1)
    predicted_data_normal = np.genfromtxt(f'predicted_data_V14_{i}.csv', delimiter=',', skip_header=1)
    
    # Calculate MSE for hippo datasets
    mse_hippo = LossFunction.mean_squared_error(actual_data_hippo, predicted_data_hippo)
    mse_all_datasets_hippo.append(mse_hippo)
    
    # Calculate MSE for normal datasets
    mse_normal = LossFunction.mean_squared_error(actual_data_normal, predicted_data_normal)
    mse_all_datasets_normal.append(mse_normal)

# Convert lists to numpy arrays for easy manipulation
mse_all_datasets_hippo = np.array(mse_all_datasets_hippo)
mse_all_datasets_normal = np.array(mse_all_datasets_normal)

# Calculate average MSE over datasets for hippo and normal
average_mse_hippo = np.mean(mse_all_datasets_hippo, axis=0)
average_mse_normal = np.mean(mse_all_datasets_normal, axis=0)

print(average_mse_hippo)

# Plot the average MSE for each dimension separately
fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Plot average MSE for normal datasets
axs[0].plot(average_mse_normal, label='Normal', color='red')
axs[1].plot(average_mse_normal, label='Normal', color='red')
axs[2].plot(average_mse_normal, label='Normal', color='red')

# Plot average MSE for hippo datasets
axs[0].plot(average_mse_hippo, label='Hippo', color='blue')
axs[1].plot(average_mse_hippo, label='Hippo', color='blue')
axs[2].plot(average_mse_hippo, label='Hippo', color='blue')

# Set titles and labels
for i, ax in enumerate(axs):
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.grid(True)
    ax.legend()

axs[-1].set_xlabel('Time Step')

plt.tight_layout()
plt.show()
