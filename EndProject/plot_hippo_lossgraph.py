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

# Plot the loss function
plt.plot(mse)
plt.xlabel('Training Iterations')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Loss Function (MSE) over Training Iterations')
plt.grid(True)
plt.show()
