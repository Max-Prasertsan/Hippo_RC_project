import numpy as np
import matplotlib.pyplot as plt

class LossFunction:
    @staticmethod
    def mean_squared_error(actual, predicted):
        return np.mean((actual - predicted) ** 2, axis=1)

# Load actual and predicted data
actual_data = np.genfromtxt('actual_data_hippo.csv', delimiter=',', skip_header=1)
predicted_data = np.genfromtxt('predicted_data_hippo.csv', delimiter=',', skip_header=1)

# Calculate Mean Squared Error (MSE)
mse = LossFunction.mean_squared_error(actual_data, predicted_data)

# Define the number of intervals
num_intervals = 10
interval_length = len(mse) // num_intervals

# Create empty lists to store MSE values for each interval
interval_mse = []

# Calculate MSE for each interval
for i in range(num_intervals):
    start_index = i * interval_length
    end_index = (i + 1) * interval_length
    interval_mse.append(mse[start_index:end_index])

# Plot the loss function as a violin plot for each interval
plt.figure(figsize=(10, 6))
plt.violinplot(interval_mse, showmeans=True, showextrema=True)
plt.xlabel('Interval')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Distribution of Mean Squared Error (MSE) in Intervals')
plt.xticks(np.arange(1, num_intervals + 1))
plt.grid(True)
plt.show()

