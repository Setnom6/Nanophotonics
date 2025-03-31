import numpy as np
import scipy.integrate as spi
from enum import Enum
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from classical_green_function.MetallicSlab import Constants


class SpectralDensityType(Enum):
    LORENTZIAN = 1
    METALLIC_SLAB = 2


class QuantumEmitter:
    def __init__(self, spectralDensityType, params, markov=False, initialCondition=1.0):
        self.spectralDensityType = spectralDensityType
        self.omega0 = params.get('omega0', 10)  # Frequency of the emitter
        self.markov = markov  # Activate or deactivate the Markov approximation
        self.initialCondition = initialCondition  # Initial condition for the probability amplitude

        if self.spectralDensityType == SpectralDensityType.LORENTZIAN:
            self.omegaA = params.get('omegaA', self.omega0)  # Central frequency of the spectral density
            self.g = params.get('g', 1)  # Coupling strength
            self.k = params.get('k', 1)  # Width of the spectral density
            self.metalSlab = None
            self.dipole = None
        elif self.spectralDensityType == SpectralDensityType.METALLIC_SLAB:
            self.metalSlab = params.get('metalSlab')
            self.dipole = np.array(params.get('dipole'))
            if self.metalSlab is None:
                raise ValueError("A MetallicSlab instance is required for this spectral density type.")
            if self.dipole is None:
                raise ValueError("A dipole vector is required for the MetallicSlab spectral density.")
        else:
            raise ValueError("Invalid spectral density type.")

    def lorentzianSpectrum(self, omega):
        """
        Lorentzian spectral density normalized to decay rate at omega0.
        """
        decayRateAtOmega0 = (self.g ** 2 / (2 * np.pi)) * self.k / (
                    (self.omega0 - self.omegaA) ** 2 + (self.k / 2) ** 2)
        spectrum = (self.g ** 2 / (2 * np.pi)) * self.k / ((omega - self.omegaA) ** 2 + (self.k / 2) ** 2)
        return spectrum / decayRateAtOmega0

    def metallicSlabSpectrum(self, omega):
        """
        Spectral density for the metallic slab using the Green's function.
        """
        self.metalSlab.setOmega(omega)  # Update frequency and recalculate dependent properties
        cutOff = 15
        eps_rel = 1e-3
        limit = 50
        G = self.metalSlab.calculateNormalizedGreenFunctionReflected(cutOff, eps_rel, limit)

        # Convert dipole moment from SI to Gaussian units (esu·cm)
        dipole_gaussian = self.dipole / np.sqrt(4 * np.pi * Constants.EPSILON_0.value)

        # Corrected spectral density formula
        prefactor = (4 * Constants.HBAR.value / Constants.C_MS.value ** 2)
        spectrum = prefactor * omega ** 2 * np.imag(dipole_gaussian.conj().T @ G @ dipole_gaussian)
        return spectrum

    def spectralDensity(self, omega):
        if self.spectralDensityType == SpectralDensityType.LORENTZIAN:
            return self.lorentzianSpectrum(omega)
        elif self.spectralDensityType == SpectralDensityType.METALLIC_SLAB:
            return self.metallicSlabSpectrum(omega)
        else:
            raise ValueError("Invalid spectral density type.")

    def kernelIntegrand(self, omega, tau):
        """
        Integrand for the memory kernel.
        """
        return self.spectralDensity(omega) * np.exp(1j * (self.omega0 - omega) * tau)

    def computeIntegralKernel(self, tau):
        """
        Numerical integration to compute the memory kernel.
        """
        # Select a large enough range for the integral
        if self.spectralDensityType == SpectralDensityType.LORENTZIAN:
            omega_max = self.omega0 + 10 * self.k  # A value large enough for the Lorentzian spectrum
        else:
            omega_max = self.omega0 * 10  # A value large enough for the metallic slab spectrum

        real_part = spi.quad(lambda x: np.real(self.kernelIntegrand(x, tau)), 0, omega_max, limit=100)[0]
        imag_part = spi.quad(lambda x: np.imag(self.kernelIntegrand(x, tau)), 0, omega_max, limit=100)[0]
        return real_part + 1j * imag_part

    def computeMarkovKernel(self):
        """
        Compute the kernel in the Markov approximation.
        """
        return np.pi * self.spectralDensity(self.omega0)

    def computeKernelArray(self, taus):
        """
        Precompute the kernel for all taus.
        """
        if self.markov:
            kernel_value = self.computeMarkovKernel()
            return np.full_like(taus, kernel_value, dtype=complex)
        else:
            return np.array([self.computeIntegralKernel(tau) for tau in taus])

    def convolutionIntegral(self, cE, kValues, dt, n):
        """
        Efficient convolution using trapezoidal integration.
        """
        integrand = kValues[:n + 1] * cE[:n + 1][::-1]
        return np.trapz(integrand, dx=dt)

    def computeProbability(self, T=10.0, dt=0.01):
        """
        Compute the probability |ce(t)|^2 over time.
        """
        N = int(T / dt)
        taus = np.linspace(0, T, N)
        kernels = self.computeKernelArray(taus)

        cE = np.zeros(N, dtype=complex)
        cE[0] = self.initialCondition # Initial condition

        for n in range(1, N):
            tN = n * dt

            if self.markov:
                # Markov approximation: simple exponential decay
                gamma = kernels[0]
                cE[n] = np.exp(-gamma * tN)
            else:
                # Non-Markovian dynamics: Runge-Kutta 4th order method
                def f(t, c):
                    return -self.convolutionIntegral(cE, kernels, dt, n)

                k1 = dt * f(tN, cE[n - 1])
                k2 = dt * f(tN + dt / 2, cE[n - 1] + k1 / 2)
                k3 = dt * f(tN + dt / 2, cE[n - 1] + k2 / 2)
                k4 = dt * f(tN + dt, cE[n - 1] + k3)

                cE[n] = cE[n - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return taus, np.abs(cE) ** 2



