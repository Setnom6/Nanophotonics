import os, sys
import numpy as np
import matplotlib.pyplot as plt
from QuantumEmitter import QuantumEmitter, SpectralDensityType
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from classical_green_function.MetallicSlab import Constants, MetallicSlab, Metals

kindOfSpectralDensity = SpectralDensityType.LORENTZIAN

# Parameters for the MetallicSlab spectral density

epsilon1 = 1.0
epsilon3 = 1.0
z = 5 * Constants.NM.value  # Distance from the source
t = 20 * Constants.NM.value  # Thickness of the slab
omega = 10*Constants.EV.value  # Silly value, just for initialization
metal = Metals.SILVER
metalSlab = MetallicSlab(metal=metal, epsilon1=epsilon1, epsilon3=epsilon3, omega=omega, t=t, z=z)
plasmon_resonance = metal.plasmaFrequency / np.sqrt(metal.epsilonB + 1) * Constants.EV.value

paramsMetallicSlab = {
    'metalSlab': metalSlab,
    'dipole': np.array([1, 0, 0])*Constants.NM.value,
    'omega0': 10*Constants.EV.value  # Frequency of the emitter
}


# Parameters for the Lorentzian spectral density

paramsLorentzian = {
    'omega0': 10*Constants.EV.value,  # Frequency of the emitter
    'omegaA': plasmon_resonance,  # Central frequency of the spectral density
    'g': 0.1 * plasmon_resonance,  # Coupling strength
    'k': 0.1*plasmon_resonance # Width of the spectral density
}

# Create QuantumEmitter instances
emitter_lorentzian = QuantumEmitter(SpectralDensityType.LORENTZIAN, paramsLorentzian)
emitter_metallic_slab = QuantumEmitter(SpectralDensityType.METALLIC_SLAB, paramsMetallicSlab)

# Frequencies for comparison
frequencies = np.linspace(0.5 * plasmon_resonance, 1.5 * plasmon_resonance, 100)

# Compute spectra
spectrum_lorentzian = [emitter_lorentzian.spectralDensity(omega) for omega in frequencies]
spectrum_metallic_slab = [emitter_metallic_slab.spectralDensity(omega) for omega in frequencies]

# Plot comparison
#plt.plot(frequencies / Constants.EV.value, spectrum_lorentzian, label='Lorentzian')
plt.plot(frequencies / Constants.EV.value, spectrum_metallic_slab, label='Metallic Slab')
plt.xlabel('Frequency (eV)')
plt.ylabel('Spectral density')
plt.legend()
plt.show()
