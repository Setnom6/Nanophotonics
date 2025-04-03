import numpy as np
import matplotlib.pyplot as plt
from QuantumEmitter import QuantumEmitter, SpectralDensityType
from classical_green_function.MetallicSlab import Constants, MetallicSlab, Metals
import os
from datetime import datetime
import time

# Parameter configuration
metal = Metals.SILVER
slabThickness = 20  # nm
emitterWavelength = 500  # nm
emitterPosition = 1  # nm
dipoleMoment = 1.0  # e·nm
plasmonResonance_eV = metal.plasmaFrequency / np.sqrt(metal.epsilonB + 1)

# Integration configuration
cutOff = [50, 10]
numPoints = 1000

# Range of dipoles (nm)
dipoles = np.array([[1, 0, 0], [1/(np.sqrt(2)), 0, 1/(np.sqrt(2))]])

# Time configuration
T = 10  # Units of 2π/ω0
dt = 0.001

# Calculate emitter frequency (eV)
emitterFrequency = 1240 / emitterWavelength  # eV
T = T / emitterFrequency
dt = dt / emitterFrequency

# Initialize storage for results
probabilities = {"simulation": [], "lorentzian": None, "markov": None}
lifetimes = {"simulation": [], "lorentzian": None, "markov": None}
globalTimes = None

# Create the metallic slab
metallicSlab = MetallicSlab(
    metal=metal,
    epsilon1=1.0,
    epsilon3=1.0,
    omega=emitterFrequency * Constants.EV.value,  # eV → s⁻¹
    t=slabThickness * Constants.NM.value,  # nm → m
    z=emitterPosition * Constants.NM.value,
)

# Start timing
startTime = time.time()

# Simulate for each dipole
for dipole in dipoles:
    # Real emitter (non-Markovian, coupled to silver)
    emitter = QuantumEmitter(
        SpectralDensityType.METALLIC_SLAB,
        params={
            'metalSlab': metallicSlab,
            'dipole': dipole * dipoleMoment,  # e·nm
            'omega0': emitterFrequency  # eV
        },
        cutOff=cutOff,
        numPoints=numPoints
    )

    # Simulations
    times, probabilities_simulation = emitter.computeProbability(T, dt)
    tau = emitter.computeLifetime(times, np.array(probabilities_simulation))

    probabilities["simulation"].append(probabilities_simulation)
    lifetimes["simulation"].append(tau)

    if globalTimes is None:
        globalTimes = times

# Lorentzian reference emitter (typical parameters for silver)
emitterLorentz = QuantumEmitter(
    SpectralDensityType.LORENTZIAN,
    params={
        'omegaA': plasmonResonance_eV,  # Same omega0 as the real emitter
        'g': 0.2 * plasmonResonance_eV,  # Typical coupling (adjust as needed)
        'k': 0.1 * plasmonResonance_eV,  # Lorentzian width (kappa)
        'omega0': emitterFrequency  # Emitter frequency
    }
)

_, probLorentzian = emitterLorentz.computeProbability(T, dt)
tauLorentzian = emitterLorentz.computeLifetime(globalTimes, np.array(probLorentzian))
probabilities["lorentzian"] = probLorentzian
lifetimes["lorentzian"] = tauLorentzian

# Markovian emitter (approximation)
emitterMarkov = QuantumEmitter(
    SpectralDensityType.METALLIC_SLAB,
    params={
        'metalSlab': metallicSlab,
        'dipole': np.array([1, 1, 1]) * dipoleMoment,  # Silly value as not taken into account
        'omega0': emitterFrequency
    },
    markov=True
)

_, probMarkov = emitterMarkov.computeProbability(T, dt)
tauMarkov = emitterMarkov.computeLifetime(globalTimes, np.array(probMarkov))
probabilities["markov"] = probMarkov
lifetimes["markov"] = tauMarkov

# Plot results
timesFseconds = globalTimes / (1.519 * emitterFrequency**2)
plt.figure(figsize=(12, 8))
for index, dipole in enumerate(dipoles):
    # Real curve (non-Markovian)
    plt.plot(
        timesFseconds,
        probabilities["simulation"][index],
        label=f'dipole = {tuple(dipole)} e nm (Simulation)'
    )
# Lorentzian reference curve
plt.plot(
    timesFseconds,
    probabilities["lorentzian"],
    'k--',
    label='Lorentzian (ref)'
)
# Markovian reference curve
plt.plot(
    timesFseconds,
    probabilities["markov"],
    'r:',
    label='Markov (ref)'
)

plt.xlabel('Time (fs)')
plt.ylabel('Probability $|c_e(t)|^2$')
plt.title(f'Decay near Silver Slab (Thickness = {slabThickness} nm, λ = {emitterWavelength} nm, z = {emitterPosition} nm)')
plt.legend()
plt.grid(True)
plt.ylim(0, 1.1)

# Save and show plot
if not os.path.exists('plots'):
    os.makedirs('plots')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'plots/probability_decay_{timestamp}.png', dpi=300, bbox_inches='tight')

# Print lifetimes
print("\nLifetimes (τ) for different models:")
for index, dipole in enumerate(dipoles):
    print(f"Dipole = {tuple(dipole)} e nm:")
    if lifetimes["simulation"][index] is not None:
        print(f"  - τ (Simulation)     = {lifetimes['simulation'][index] / (1.519 * emitterFrequency**2):.4f} fs")
if lifetimes["lorentzian"] is not None:
    print(f"  - τ (Lorentzian ref) = {lifetimes['lorentzian'] / (1.519 * emitterFrequency**2):.4f} fs")
if lifetimes["markov"] is not None:
    print(f"  - τ (Markov ref)     = {lifetimes['markov'] / (1.519 * emitterFrequency**2):.4f} fs")

# Print execution time
elapsedTime = time.time() - startTime
hours, rem = divmod(elapsedTime, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

plt.show()