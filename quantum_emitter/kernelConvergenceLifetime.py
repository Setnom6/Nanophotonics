import numpy as np
import matplotlib.pyplot as plt
from QuantumEmitter import QuantumEmitter, SpectralDensityType
from classical_green_function.MetallicSlab import Constants, MetallicSlab, Metals
import time
import os

startTime = time.time()

# Fixed parameters
metal = Metals.SILVER
slabThickness = 20  # nm
emitterWavelength = 300  # nm
dipoleMoment = 1.0  # e·nm
distance = 200  # nm

# Emitter frequency
emitterFrequency = 1240 / emitterWavelength  # eV
physDistance = distance * Constants.NM.value  # nm → m

# Integration configuration
numPointsValues = [500]
cutOffValues = [2, 3, 5, 7, 10]

# Matrices to store results
lifetimes = np.zeros((len(numPointsValues), len(cutOffValues)))

# Create the metallic slab
metallicSlab = MetallicSlab(
    metal=metal,
    epsilon1=1.0,
    epsilon3=1.0,
    omega=emitterFrequency * Constants.EV.value,  # eV → s⁻¹
    t=slabThickness * Constants.NM.value,  # nm → m
    z=physDistance
)

# Convergence test loop
for i, numPoints in enumerate(numPointsValues):
    for j, cutOff in enumerate(cutOffValues):
        # Create the emitter
        emitter = QuantumEmitter(
            SpectralDensityType.METALLIC_SLAB,
            params={
                'metalSlab': metallicSlab,
                'dipole': np.array([1, 1, 1]) * dipoleMoment,  # e·nm
                'omega0': emitterFrequency  # eV
            },
            cutOff=[50, cutOff],
            numPoints=numPoints
        )

        # Compute lifetime
        times, probabilities = emitter.computeProbability(10 / emitterFrequency, 0.001 / emitterFrequency)
        tau = emitter.computeLifetime(times, probabilities)

        # Store result
        lifetimes[i, j] = tau / (1.519 * emitterFrequency ** 2)  # Convert to fs

# Plot the results
plt.figure(figsize=(8, 6))
for i, numPoints in enumerate(numPointsValues):
    plt.plot(cutOffValues, lifetimes[i], marker='o', label=f'numPoints = {numPoints}')

plt.xlabel('cutOff[1]')
plt.ylabel('Lifetime (fs)')
plt.title('Convergence of Lifetime Calculation')
plt.legend()
plt.grid(True)

# Save the plot
if not os.path.exists('plots'):
    os.makedirs('plots')
timestamp = time.strftime("%Y%m%d_%H%M%S")
plt.savefig(f'plots/convergence_lifetime_{timestamp}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'plots/convergence_lifetime_{timestamp}.pdf', bbox_inches='tight')

# Print execution time
elapsedTime = time.time() - startTime
hours, rem = divmod(elapsedTime, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")