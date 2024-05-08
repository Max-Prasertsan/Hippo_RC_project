import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load actual and predicted data
actual_data = np.genfromtxt('actual_data_hippo.csv', delimiter=',', skip_header=1)
predicted_data = np.genfromtxt('predicted_data_hippo.csv', delimiter=',', skip_header=1)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot actual data
ax.plot(actual_data[:,0], actual_data[:,1], actual_data[:,2], label='Actual', color='b', lw=0.5)

# Plot predicted data
ax.plot(predicted_data[:,0], predicted_data[:,1], predicted_data[:,2], label='Predicted', color='r', lw=0.5)

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Comparison of Actual and Predicted Data against Lorenz Attractor')
ax.legend()

plt.show()
