import numpy as np
import matplotlib.pyplot as plt
from QuantumEmitter import QuantumEmitter, SpectralDensityType
from classical_green_function.MetallicSlab import Constants, MetallicSlab, Metals
import os
from datetime import datetime


def savePlot(fig, filename):
    """Saves the plot in the 'plots' folder."""
    if not os.path.exists('plots'):
        os.makedirs('plots')
    fig.savefig(f'plots/{filename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'plots/{filename}.pdf', bbox_inches='tight')


# Fixed parameters
METAL = Metals.SILVER
DISTANCE_NM = 5  # Fixed distance in nm
SLAB_THICKNESS_NM = 20  # Slab thickness in nm
EMITTER_WAVELENGTH_NM = 300  # Emitter wavelength in nm
DIPOLE_MOMENT = 1.0  # Dipole moment in e·nm

# Time configuration
T = 10  # Total time in units of 2π/ω₀
DT = 0.001  # Time step

# Dipole orientations to compare
DIPOLE_ORIENTATIONS = {
    'x': np.array([1, 0, 0]),  # Parallel to the surface
    'y': np.array([0, 1, 0]),  # Parallel to the surface
    'z': np.array([0, 0, 1])  # Perpendicular to the surface
}

# Conversion to SI units
distance = DISTANCE_NM * Constants.NM.value
slabThickness = SLAB_THICKNESS_NM * Constants.NM.value
emitterEnergy_eV = (2 * np.pi * Constants.HBAR.value * Constants.C_MS.value) / (
        EMITTER_WAVELENGTH_NM * Constants.NM.value * Constants.E_CHARGE.value)
emitterFrequency = emitterEnergy_eV * Constants.EV.value  # in s⁻¹

# Create the metallic slab
metallicSlab = MetallicSlab(
    metal=METAL,
    epsilon1=1.0,
    epsilon3=1.0,
    omega=emitterFrequency,
    t=slabThickness,
    z=distance
)

# Simulate for each orientation
results = {}
for orientationName, orientation in DIPOLE_ORIENTATIONS.items():
    emitter = QuantumEmitter(
        SpectralDensityType.METALLIC_SLAB,
        params={
            'metalSlab': metallicSlab,
            'dipole': orientation * DIPOLE_MOMENT,
            'omega0': emitterEnergy_eV
        },
        cutOff=20,
        numPoints=100
    )

    times, probabilities = emitter.computeProbability(T, DT)
    results[orientationName] = {
        'times': times,
        'probabilities': probabilities
    }

# Plot
plt.figure(figsize=(10, 6))
colors = {'x': 'red', 'y': 'green', 'z': 'blue'}
for orientationName, data in results.items():
    plt.plot(
        data['times'] * (2 * np.pi / emitterEnergy_eV),
        data['probabilities'],
        color=colors[orientationName],
        label=f'Dipole {orientationName}'
    )

plt.xlabel('Time ($2\pi/\omega_0$)')
plt.ylabel('Probability $|c_e(t)|^2$')
plt.title(
    f'Decay near silver slab ({DISTANCE_NM} nm)\n'
    f'Thickness: {SLAB_THICKNESS_NM} nm, $\lambda$: {EMITTER_WAVELENGTH_NM} nm'
)
plt.legend()
plt.grid(True)
plt.ylim(0, 1.1)

# Save and show
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
savePlot(plt.gcf(), f'dipole_orientation_comparison_{timestamp}')
plt.show()