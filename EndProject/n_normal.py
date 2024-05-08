import numpy as np
import matplotlib.pyplot as plt

# Load actual data
actual_data = np.genfromtxt('actual_data.csv', delimiter=',', skip_header=1)

# Define number of predicted datasets
n_predicted_datasets = 50

# Create subplots
plt.figure(figsize=(12, 8))

# Loop over each dimension (X, Y, Z)
for dim in range(3):
    # Get the actual data for the current dimension
    actual_dim_data = actual_data[:, dim]
    
    # Plot actual data
    plt.subplot(3, 1, dim + 1)
    plt.plot(actual_dim_data, label=f'Actual {chr(88 + dim)}', color='b', lw=0.8)

    # Loop over each predicted dataset
    for i in range(n_predicted_datasets):
        # Load predicted data for the current dataset
        predicted_data = np.genfromtxt(f'predicted_data_{i}.csv', delimiter=',', skip_header=1)
        predicted_dim_data = predicted_data[:, dim]
        
        # Plot predicted data for the current dimension
        plt.plot(predicted_dim_data, '--', lw=0.8)

    plt.ylabel(f'{chr(88 + dim)} Value')
    plt.title(f'Comparison of Actual and Predicted Data against Timestep ({chr(88 + dim)} Dimension)')
    plt.legend()
    plt.grid(True)

plt.xlabel('Timestep')
plt.tight_layout()
plt.show()
