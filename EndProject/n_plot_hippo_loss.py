import numpy as np
import matplotlib.pyplot as plt

class LossFunction:
    @staticmethod
    def mean_squared_error(actual, predicted):
        return np.mean((actual - predicted) ** 2, axis=1)

# Define number of datasets
n = 50

# Initialize array to store MSE for each dataset
mse_all_datasets = []


# Load actual and predicted data for each dataset and compute the MSE
for i in range(n):
    actual_data = np.genfromtxt(f'actual_data_hippo_V1_{i}.csv', delimiter=',', skip_header=1)
    predicted_data = np.genfromtxt(f'predicted_data_hippo_V1_{i}.csv', delimiter=',', skip_header=1)
    
    # Calculate Mean Squared Error (MSE)
    mse = LossFunction.mean_squared_error(actual_data, predicted_data)
    mse_all_datasets.append(mse)

# Convert list to numpy array for easy manipulation
mse_all_datasets = np.array(mse_all_datasets)

# Calculate average MSE over datasets
average_mse = np.mean(mse_all_datasets, axis=0)

# Plot the average loss function
plt.plot(average_mse)
plt.xlabel('Training Iterations')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Average Loss Function (MSE) over Training Iterations')
plt.grid(True)
plt.show()
