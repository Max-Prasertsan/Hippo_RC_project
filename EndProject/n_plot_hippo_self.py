import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define number of datasets
n = 25

# Initialize dictionaries to store aggregated actual and predicted data for each timestep
aggregated_actual_data = {}
aggregated_predicted_data = {}

# Loop through each dataset
for i in range(n):
    # Load actual and predicted data
    actual_data = np.genfromtxt(f'actual_data_hippo_{i}.csv', delimiter=',', skip_header=1)
    predicted_data = np.genfromtxt(f'predicted_data_hippo_{i}.csv', delimiter=',', skip_header=1)

    # Aggregate data
    for j in range(len(actual_data)):
        timestep = tuple(actual_data[j])
        if timestep in aggregated_actual_data:
            aggregated_actual_data[timestep].append(actual_data[j])
        else:
            aggregated_actual_data[timestep] = [actual_data[j]]
            
    for j in range(len(predicted_data)):
        timestep = tuple(predicted_data[j])
        if timestep in aggregated_predicted_data:
            aggregated_predicted_data[timestep].append(predicted_data[j])
        else:
            aggregated_predicted_data[timestep] = [predicted_data[j]]

# Compute the average for each timestep
average_actual_data = np.array([np.mean(aggregated_actual_data[key], axis=0) for key in aggregated_actual_data])
average_predicted_data = np.array([np.mean(aggregated_predicted_data[key], axis=0) for key in aggregated_predicted_data])

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot aggregated actual data
ax.plot(average_actual_data[:,0], average_actual_data[:,1], average_actual_data[:,2], label='Average Actual', color='b')

# Plot aggregated predicted data
ax.plot(average_predicted_data[:,0], average_predicted_data[:,1], average_predicted_data[:,2], label='Average Predicted', color='r')

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Comparison of Average Actual and Predicted Data against Lorenz Attractor')
ax.legend()

plt.show()

