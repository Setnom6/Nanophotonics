import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from QuantumEmitter import QuantumEmitter, SpectralDensityType
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from classical_green_function.MetallicSlab import Constants, MetallicSlab, Metals

import time  # To measure execution time

# Start timing
start_time = time.time()

# Parameters for the simulation
kindOfSpectralDensity = SpectralDensityType.LORENTZIAN
cutOff = 10  # Cut-off parameter for integration
numPoints = 50  # Number of points for Gauss-Legendre quadrature

# Parameters for the MetallicSlab spectral density
z = 5  # Distance
t = 20  # Thickness of the slab
metal = Metals.SILVER
metallicSlab = MetallicSlab(metal=metal, epsilon1=1.0, epsilon3=1.0,
                            omega=10, t=t, z=z)
plasmonResonance = (metal.plasmaFrequency / np.sqrt(metal.epsilonB + 1))
omega0 = plasmonResonance  # Set to plasmon resonance frequency
dipoleMoment = 1.6e-28 / Constants.NM.value  # ~1 e·nm in C·m

paramsMetallicSlab = {
    'metalSlab': metallicSlab,
    'dipole': np.array([1, 0, 0]) * dipoleMoment,
    'omega0': omega0
}

# Parameters for Lorentzian with comparable parameters
paramsLorentzian = {
    'omega0': omega0,
    'omegaA': omega0,  # Center at same frequency
    'g': 0.1 * omega0,
    'k': 0.1 * omega0
}

# Create and precompute
emitterLorentzian = QuantumEmitter(SpectralDensityType.LORENTZIAN, paramsLorentzian, cutOff=cutOff, numPoints=numPoints)
emitterMetallic = QuantumEmitter(SpectralDensityType.METALLIC_SLAB, paramsMetallicSlab, cutOff=cutOff, numPoints=numPoints)

# Times for comparison
T = 50 / plasmonResonance  # Final time
dt = 0.01 / plasmonResonance  # Time step

# Compute probabilities
timesLorentzian, probabilitiesLorentzian = emitterLorentzian.computeProbability(T, dt)
timesMetallic, probabilitiesMetallicSlab = emitterMetallic.computeProbability(T, dt)

# Plot comparison
plt.plot(timesLorentzian, probabilitiesLorentzian, label='Lorentzian')
plt.plot(timesMetallic, probabilitiesMetallicSlab, label='Metallic Slab')

plt.xlabel('Time')
plt.ylabel('Probability')

plt.legend()

# End timing
end_time = time.time()

# Format elapsed time as h:min:s
elapsedTime = end_time - start_time
hours, rem = divmod(elapsedTime, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

plt.show()