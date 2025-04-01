import numpy as np
import matplotlib.pyplot as plt
from QuantumEmitter import QuantumEmitter, SpectralDensityType
from classical_green_function.MetallicSlab import Constants, MetallicSlab, Metals
import os
from datetime import datetime
import time

# Start timing
start_time = time.time()


def savePlotWithTimestamp(figure, baseName="spectral_density"):
    """Save plot to a 'plots' folder with timestamp"""
    if not os.path.exists('plots'):
        os.makedirs('plots')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figure.savefig(f'plots/{baseName}_{timestamp}.png', dpi=300, bbox_inches='tight')
    figure.savefig(f'plots/{baseName}_{timestamp}.pdf', bbox_inches='tight')


# Parameter configuration (using nm and eV)
metal = Metals.SILVER
slabThickness_nm = 20  # Thickness in nm
emitterWavelength_nm = 300  # Wavelength in nm
dipoleMoment = 1.0  # in e·nm

# Integration configuration
cutOff = 20
num_points = 100

# Convert to SI units for MetallicSlab
slabThickness = slabThickness_nm * Constants.NM.value
emitterWavelength = emitterWavelength_nm * Constants.NM.value

# Calculate frequencies
emitterEnergy_eV = (2 * np.pi * Constants.HBAR.value * Constants.C_MS.value) / (
            emitterWavelength * Constants.E_CHARGE.value)
emitterFrequency_SI = emitterEnergy_eV * Constants.EV.value  # in s^-1 for MetallicSlab

# Calculate plasmon resonance frequency (in eV)
plasmonResonance_eV = metal.plasmaFrequency / np.sqrt(metal.epsilonB + 1)

# Range of distances to explore (in nm)
distances_nm = np.array([1, 5, 10])  # in nm
distances = distances_nm * Constants.NM.value  # convert to m for MetallicSlab

# Range of frequencies to evaluate (in eV)
frequencies_eV = np.linspace(0.1 * emitterEnergy_eV, 3 * emitterEnergy_eV, 300)
maxSpectralDensity = []

plt.figure(figsize=(12, 8))

for distance_nm, distance in zip(distances_nm, distances):
    metallicSlab = MetallicSlab(metal=metal, epsilon1=1.0, epsilon3=1.0,
                                omega=emitterFrequency_SI, t=slabThickness, z=distance)

    params = {
        'metalSlab': metallicSlab,
        'dipole': np.array([1, 1, 1]) * dipoleMoment,  # in e·nm
        'omega0': emitterEnergy_eV  # in eV
    }

    emitter = QuantumEmitter(SpectralDensityType.METALLIC_SLAB, params,
                             cutOff=cutOff, numPoints=num_points)

    spectralDensity = [emitter.spectralDensity(omega_eV) for omega_eV in frequencies_eV]

    # Plot with distance in wavelength units
    label = f'distance = {distance_nm / emitterWavelength_nm:.4f}λ'
    plt.plot(frequencies_eV, spectralDensity, label=label)
    maxSpectralDensity.append(np.max(spectralDensity))

# Add vertical line at plasmon resonance
plt.axvline(x=plasmonResonance_eV, color='gray', linestyle='--',
            label=f'Surface Plasmon Resonance\n({plasmonResonance_eV:.2f} eV)')

plt.xlabel('Frequency (eV)')
plt.ylabel('Normalized Spectral Density (Γ/Γ₀)')
plt.title(
    f'Normalized Spectral Density near Silver Slab\n(Thickness = {slabThickness_nm:.1f} nm, λ = {emitterWavelength_nm:.1f} nm)')
plt.legend()
plt.grid(True)

# Set y-axis limits
plt.ylim([0, np.max(maxSpectralDensity)])

# Save and show plot
savePlotWithTimestamp(plt.gcf())

# End timing
end_time = time.time()
elapsedTime = end_time - start_time
hours, rem = divmod(elapsedTime, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

plt.show()