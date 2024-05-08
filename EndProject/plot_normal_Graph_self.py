import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Read the predicted and actual data from the CSV files
predicted_df = pd.read_csv('predicted_data.csv')
actual_df = pd.read_csv('actual_data.csv')

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Actual data
x_actual = actual_df['x']
y_actual = actual_df['y']
z_actual = actual_df['z']
ax.plot(x_actual, y_actual, z_actual, label='Actual', color='blue', alpha=0.7)

# Predicted data
x_pred = predicted_df['x']
y_pred = predicted_df['y']
z_pred = predicted_df['z']
ax.plot(x_pred, y_pred, z_pred, label='Predicted', color='orange', alpha=0.7)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor: Actual vs Predicted')
ax.legend()

plt.show()