# Program 1: Plot Predicted vs Actual X, Y, Z

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read predicted and actual data from CSV files
predicted_df = pd.read_csv('predicted_data.csv')
actual_df = pd.read_csv('actual_data.csv')

# Generate time steps
timesteps = np.arange(len(actual_df))

# Create subplots
plt.figure(figsize=(12, 8))

# Plot X dimension
plt.subplot(3, 1, 1)
plt.plot(timesteps, actual_df['x'], label='Actual X', color='b', lw=0.8)
plt.plot(timesteps, predicted_df['x'], '--', label='Predicted X', color='black', lw=0.8)
plt.ylabel('X Value')
plt.title('Comparison of Actual and Predicted Data against Timestep')
plt.legend()
plt.grid(True)

# Plot Y dimension
plt.subplot(3, 1, 2)
plt.plot(timesteps, actual_df['y'], label='Actual Y', color='g', lw=0.8)
plt.plot(timesteps, predicted_df['y'], '--', label='Predicted Y', color='black', lw=0.8)
plt.ylabel('Y Value')
plt.legend()
plt.grid(True)

# Plot Z dimension
plt.subplot(3, 1, 3)
plt.plot(timesteps, actual_df['z'], label='Actual Z', color='r', lw=0.8)
plt.plot(timesteps, predicted_df['z'], '--', label='Predicted Z', color='black', lw=0.8)
plt.xlabel('Timestep')
plt.ylabel('Z Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

