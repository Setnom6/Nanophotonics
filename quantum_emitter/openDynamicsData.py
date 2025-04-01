import json
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open('./data/simulation_data_20250401_184355.json', 'r') as f:
    data = json.load(f)

# Convert back to numpy arrays
distances = np.array(data["distances"])
times = [np.array(t) for t in data["times"]]
probabilities = [np.array(p) for p in data["probabilities"]]

# Plot
plt.figure(figsize=(10, 6))
for i, d in enumerate(distances):
    plt.plot(times[i], probabilities[i], label=f'z = {d} nm')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.legend()
plt.show()