import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp
import csv

class ReservoirComputer:
    def __init__(self, dim_system, dim_reservoir, rho, sigma):
        self.dim_system = dim_system
        self.dim_reservoir = dim_reservoir
        self.r_state = np.zeros(dim_reservoir)
        self.A = self.generate_adjacency_matrix(dim_reservoir, rho, sigma)
        self.W_in = self.generate_input_weights(dim_reservoir, dim_system)
        self.W_out = np.zeros((dim_system, dim_reservoir))
    
    def advance_r_state(self, u):
        self.r_state = self.sigmoid(np.dot(self.A, self.r_state) + np.dot(self.W_in, u))
        return self.r_state
    
    def v(self):
        return np.dot(self.W_out, self.r_state)
    
    def train(self, trajectory):
        R = np.zeros((self.dim_reservoir, trajectory.shape[0]))
        for i in range(trajectory.shape[0]):
            R[:, i] = self.r_state
            u = trajectory[i]
            self.advance_r_state(u)
        self.W_out = self.linear_regression(R, trajectory)
    
    def predict(self, steps):
        prediction = np.zeros((steps, self.dim_system))
        for i in range(steps):
            v = self.v()
            prediction[i] = v
            self.advance_r_state(prediction[i])
        return prediction

    @staticmethod
    def generate_input_weights(dim_reservoir, dim_system):
        # Generate a connectivity pattern where each reservoir neuron connects to a subset of system neurons
        # Here, we connect each reservoir neuron to a fixed number of system neurons
        W_in = np.zeros((dim_reservoir, dim_system))
        for i in range(dim_reservoir):
            connected_neurons = np.random.choice(dim_system, size=min(5, dim_system), replace=False)
            num_excitatory = int(0.8 * len(connected_neurons))
            excitatory_neurons = np.random.choice(connected_neurons, size=num_excitatory, replace=False)
            inhibitory_neurons = np.setdiff1d(connected_neurons, excitatory_neurons)
            # Assign excitatory weights to all connected neurons
            W_in[i, connected_neurons] = np.random.uniform(-1, 1, len(connected_neurons))
            # Correct the weights for inhibitory neurons
            W_in[i, inhibitory_neurons] *= -1  # Ensure inhibitory neurons have negative weights
        return W_in
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def generate_adjacency_matrix(dim_reservoir, rho, sigma):
        graph = nx.gnp_random_graph(dim_reservoir, sigma)
        graph = nx.to_numpy_array(graph)
        random_array = 2 * (np.random.rand(dim_reservoir, dim_reservoir) - 0.5)
        rescaled = graph * random_array
        return ReservoirComputer.scale_matrix(rescaled, rho)

    @staticmethod
    def scale_matrix(A, rho):
        eigenvalues, _ = np.linalg.eig(A)
        max_eigenvalue = np.amax(eigenvalues)
        A = A / np.absolute(max_eigenvalue) * rho
        return A

    @staticmethod
    def linear_regression(R, trajectory, beta=0.0001):  
        Rt = np.transpose(R)
        inverse_part = np.linalg.inv(np.dot(R, Rt) + beta * np.identity(R.shape[0]))
        return np.dot(np.dot(trajectory.T, Rt), inverse_part)


dim_system = 3
dim_reservoir = 828

rho = 3.635714285714286

sigma = 1.5142857142857145

# Generate Rössler attractor data
def rossler(t, xyz, a=0.2, b=0.2, c=5.7):
    x, y, z = xyz
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

t_span = (0, 100)
size = 10000
t_eval = np.linspace(*t_span, size)
initial_conditions = [0.1, 0.1, 0.1]  # Initial condition for Rössler attractor
solution = solve_ivp(rossler, t_span, initial_conditions, t_eval=t_eval)

# Extract and normalize data
data = solution.y.T
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)
data_normalized = (data - data_mean) / data_std

# Use part of the data for training and part for validation
train_data = data_normalized[:8000]
valid_data = data_normalized[8000:]

# Create and train reservoir computer
rc = ReservoirComputer(dim_system, dim_reservoir, rho, sigma)
rc.train(train_data)

# Predict using the trained reservoir computer
predicted_data = rc.predict(valid_data.shape[0])

# Save the predicted data to a CSV file
with open('predicted_data_rossler.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['x', 'y', 'z'])
    for row in predicted_data:
        writer.writerow(row)

# Save the actual data to a CSV file
with open('actual_data_rossler.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['x', 'y', 'z'])
    for row in valid_data:
        writer.writerow(row)
