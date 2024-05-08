import numpy as np
import matplotlib.pyplot as plt

# Load actual and predicted data
actual_data = np.genfromtxt('actual_data_hippo.csv', delimiter=',', skip_header=1)
predicted_data = np.genfromtxt('predicted_data_hippo.csv', delimiter=',', skip_header=1)

# Calculate the absolute difference between actual and predicted data
difference = np.abs(actual_data - predicted_data)

# Plot the difference for each dimension separately
fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

for i, ax in enumerate(axs):
    ax.plot(difference[:, i], label=f'Dimension {"XYZ"[i]}')
    ax.set_ylabel('Absolute Difference')
    ax.set_title(f'Absolute Difference for Dimension {"XYZ"[i]}')  # Change title
    ax.grid(True)
    ax.legend()

axs[-1].set_xlabel('Time Step')

plt.tight_layout()
plt.show()



