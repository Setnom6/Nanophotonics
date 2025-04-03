import numpy as np
import matplotlib.pyplot as plt
from QuantumEmitter import QuantumEmitter, SpectralDensityType
from classical_green_function.MetallicSlab import Constants, MetallicSlab, Metals
import os
from datetime import datetime
import time
import json


def saveDataToJson(dataDict, baseName="simulation_data"):
    """
    Save data to a JSON file with a timestamp.
    """
    if not os.path.exists('data'):
        os.makedirs('data')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filePath = f'data/{baseName}_{timestamp}.json'

    # Convert numpy arrays to lists for JSON compatibility
    jsonData = {
        "distances": dataDict["distances"].tolist(),
        "times": [t.tolist() for t in dataDict["times"]],
        "probabilities": [p.tolist() for p in dataDict["probabilities"]],
        "parameters": dataDict["parameters"]
    }

    with open(filePath, 'w') as f:
        json.dump(jsonData, f, indent=4)
    print(f"Data saved to {filePath}")


# Parameter configuration
metal = Metals.SILVER
slabThickness = 20  # nm
emitterWavelength = 500  # nm
dipoleMoment = 1.0  # e·nm
plasmonResonance_eV = metal.plasmaFrequency / np.sqrt(metal.epsilonB + 1)

# Integration configuration
cutOff = [50, 5]
numPoints = 500

# Range of distances (nm)
distances = np.concatenate([
    np.array([0.1, 1, 5, 10, 20]),  # Short distances
    np.array([0.5, 0.75, 1.0, 2.0]) * emitterWavelength # Long distances
])

distances = np.array([0.5, 0.75, 1.0, 2.0]) * emitterWavelength   # Short distances

# Time configuration
T = 30  # Units of 2π/ω0
dt = 0.001

# Calculate emitter frequency (eV)
emitterFrequency = 1240 / emitterWavelength  # eV
T = T / emitterFrequency
dt = dt / emitterFrequency

# Initialize results storage
results = {
    "distances": [],
    "times": [],
    "probabilities": [],
    "probabilities_lorentz": [],
    "probabilities_markov": [],
    "lifetimes" : [],
    "lifetimes_lorentz" : [],
    "lifetimes_markov" : [],
    "parameters": {
        "slabThickness": slabThickness,
        "emitterWavelength": emitterWavelength,
        "emitterFrequency": emitterFrequency,
        "dipoleMoment": dipoleMoment,
        "T": T,
        "dt": dt
    }
}

# Simulation loop
startTime = time.time()
for distance in distances:
    physDistance = distance * Constants.NM.value  # nm → m
    metallicSlab = MetallicSlab(
        metal=metal,
        epsilon1=1.0,
        epsilon3=1.0,
        omega=emitterFrequency * Constants.EV.value,  # eV → s⁻¹
        t=slabThickness * Constants.NM.value,  # nm → m
        z=physDistance
    )

    # Real emitter (non-Markovian, coupled to silver)
    emitter = QuantumEmitter(
        SpectralDensityType.METALLIC_SLAB,
        params={
            'metalSlab': metallicSlab,
            'dipole': np.array([1, 1, 1]) * dipoleMoment,  # e·nm
            'omega0': emitterFrequency  # eV
        },
        cutOff=cutOff,
        numPoints=numPoints
    )

    # Lorentzian reference emitter (typical parameters for silver)
    emitter_lorentz = QuantumEmitter(
        SpectralDensityType.LORENTZIAN,
        params={
            'omegaA': plasmonResonance_eV,  # Same omega0 as the real emitter
            'g': 0.2 * plasmonResonance_eV,  # Typical coupling (Normalization cancels it)
            'k': 0.5 * plasmonResonance_eV,  # Lorentzian width (kappa)
            'omega0': emitterFrequency  # Emitter frequency
        }
    )

    # Markovian emitter (approximation)
    emitter_markov = QuantumEmitter(
        SpectralDensityType.METALLIC_SLAB,
        params={
            'metalSlab': metallicSlab,
            'dipole': np.array([1, 1, 1]) * dipoleMoment,
            'omega0': emitterFrequency
        },
        markov=True
    )

    # Simulations
    times, probabilities = emitter.computeProbability(T, dt)
    _, probabilities_lorentz = emitter_lorentz.computeProbability(T, dt)
    _, probabilities_markov = emitter_markov.computeProbability(T, dt)

    # Store results
    results["distances"].append(distance)
    results["times"].append(times)
    results["probabilities"].append(np.maximum(probabilities, 0))
    results["probabilities_lorentz"].append(probabilities_lorentz)  # New entry
    results["probabilities_markov"].append(probabilities_markov)    # New entry
    tau = emitter.computeLifetime(times, probabilities)
    tau_lorentz = emitter_lorentz.computeLifetime(times, probabilities_lorentz)
    tau_markov = emitter_markov.computeLifetime(times, probabilities_markov)

    results["lifetimes"].append(tau)
    results["lifetimes_lorentz"].append(tau_lorentz)
    results["lifetimes_markov"].append(tau_markov)

# Convert to numpy arrays
results["distances"] = np.array(results["distances"])
results["times"] = np.array(results["times"])
results["probabilities"] = np.array(results["probabilities"])
results["probabilities_lorentz"] = np.array(results["probabilities_lorentz"])
results["probabilities_markov"] = np.array(results["probabilities_markov"])
results["lifetimes"] = np.array(results["lifetimes"])
results["lifetimes_lorentz"] = np.array(results["lifetimes_lorentz"])
results["lifetimes_markov"] = np.array(results["lifetimes_markov"])

# Save data to JSON
#saveDataToJson(results)

# Plot results
plt.figure(figsize=(12, 8))
for i, distance in enumerate(results["distances"]):
    times_fseconds = results["times"][i] / (1.519 * emitterFrequency**2)
    # Real curve (non-Markovian)
    plt.plot(
        times_fseconds,
        results["probabilities"][i],
        label=f'z = {distance / emitterWavelength:.5f}λ (Simulation)'
    )

plt.xlabel('Time (fs)')
plt.ylabel('Probability $|c_e(t)|^2$')
plt.title(f'Decay near Silver Slab (Thickness = {slabThickness} nm, λ = {emitterWavelength} nm)')
plt.legend()
plt.grid(True)
plt.ylim(0, 1.1)

times_fseconds = results["times"][0] / (1.519 * emitterFrequency**2)
# Markovian reference curve
plt.plot(
        times_fseconds,
        results["probabilities_markov"][0],
        'r:',
        label='Markov (ref)' if i == 0 else ""
    )

# Save and show plot
if not os.path.exists('plots'):
    os.makedirs('plots')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'plots/probability_decay_long_distances_{timestamp}.pdf', bbox_inches='tight')

# Lorentzian reference curve
plt.plot(
    times_fseconds,
    results["probabilities_lorentz"][0],
        'k--',
        label='Lorentzian (ref)' if i == 0 else ""
)

# Print lifetimes
print("\nLifetimes (τ) for different models:")
for i, distance in enumerate(results["distances"]):
    print(f"Distance z = {distance:.2f} nm:")
    if results['lifetimes'][i] is not None:
        print(f"  - τ (Simulation)     = {results['lifetimes'][i] / (1.519 * emitterFrequency**2):.4f} fs")
    if results['lifetimes_lorentz'][i] is not None:
        print(f"  - τ (Lorentzian ref) = {results['lifetimes_lorentz'][i] / (1.519 * emitterFrequency**2):.4f} fs")
    if results['lifetimes_markov'][i] is not None:
        print(f"  - τ (Markov ref)     = {results['lifetimes_markov'][i] / (1.519 * emitterFrequency**2):.4f} fs")

# Print execution time
elapsedTime = time.time() - startTime
hours, rem = divmod(elapsedTime, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

plt.show()