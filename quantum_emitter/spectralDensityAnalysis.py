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
emitterWavelength_nm = 500  # Wavelength in nm
dipoleMoment = 1.0  # in e·nm
dipole = np.array([1, 1, 1])

# Integration configuration
cutOff = [50, 10]
numPoints = 1000

# Convert to SI units for MetallicSlab
slabThickness = slabThickness_nm * Constants.NM.value
emitterWavelength = emitterWavelength_nm * Constants.NM.value

# Calculate emitter frequency (eV)
emitterFrequency = 1240 / emitterWavelength_nm  # eV

# Calculate plasmon resonance frequency (in eV)
plasmonResonance_eV = metal.plasmaFrequency / np.sqrt(metal.epsilonB + 1)

# Range of distances to explore (in nm)
distances = np.concatenate([
    np.array([0.1, 1, 5, 10, 20]),  # Short distances
    np.array([0.5, 0.75, 1.0, 2.0]) * emitterWavelength_nm  # Long distances
])

distances = np.array([0.1, 1, 5, 10, 20])

# Range of frequencies to evaluate (in eV)
frequencies_eV = np.linspace(0.1 * emitterFrequency, 2 * emitterFrequency, 300)
maxSpectralDensity = []

plt.figure(figsize=(12, 8))

for distance in distances:
    metallicSlab = MetallicSlab(metal=metal, epsilon1=1.0, epsilon3=1.0,
                                omega=emitterFrequency * Constants.EV.value,  # eV → s⁻¹
                                t=slabThickness, z=distance*Constants.NM.value)

    params = {
        'metalSlab': metallicSlab,
        'dipole': dipole * dipoleMoment,  # in e·nm
        'omega0': emitterFrequency  # in eV
    }

    emitter = QuantumEmitter(SpectralDensityType.METALLIC_SLAB, params,
                             cutOff=cutOff, numPoints=numPoints)

    spectralDensity = [emitter.spectralDensity(omega_eV) for omega_eV in frequencies_eV]

    # Plot with distance in wavelength units
    label = f'distance = {distance / emitterWavelength_nm:.4f}λ'
    plt.plot(frequencies_eV, spectralDensity, label=label)
    maxSpectralDensity.append(np.max(spectralDensity))

# Add vertical line at plasmon resonance
plt.axvline(x=plasmonResonance_eV, color='gray', linestyle='--',
            label=f'Surface Plasmon Resonance\n({plasmonResonance_eV:.2f} eV)')

plt.xlabel('Frequency (eV)')
plt.ylabel('Normalized Spectral Density (Γ/Γ₀)')
plt.title(
    f'Normalized Spectral Density near Silver Slab\n(Thickness = {slabThickness_nm:.1f} nm, λ = {emitterWavelength_nm:.1f} nm, dipole = {tuple(dipole)} enm)')
plt.legend()
plt.grid(True)

# Filter out NaN values from maxSpectralDensity
maxSpectralDensity = np.array(maxSpectralDensity)
maxSpectralDensity = maxSpectralDensity[~np.isnan(maxSpectralDensity)]  # Remove NaN values

# Set y-axis limits
if len(maxSpectralDensity) > 0:  # Ensure there are valid values
    plt.ylim([0, np.max(maxSpectralDensity)])
# Set y-axis limits
plt.ylim([0, np.max(maxSpectralDensity)])

# Save and show plot
savePlotWithTimestamp(plt.gcf(), "spectral_density_short_distances")

# End timing
end_time = time.time()
elapsedTime = end_time - start_time
hours, rem = divmod(elapsedTime, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

plt.show()