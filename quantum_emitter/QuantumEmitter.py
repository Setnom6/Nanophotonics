import numpy as np
import scipy.integrate as spi
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from enum import Enum
import sys
import os
from joblib import Parallel, delayed
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from classical_green_function.MetallicSlab import Constants


class SpectralDensityType(Enum):
    LORENTZIAN = 1
    METALLIC_SLAB = 2


class QuantumEmitter:
    def __init__(self, spectralDensityType, params, markov=False, initialCondition=1.0, cutOff=10, numPoints=50):
        self.spectralDensityType = spectralDensityType
        self.omega0 = params.get('omega0', 10 * Constants.EV.value)  # Frequency of the emitter
        self.markov = markov  # Activate or deactivate the Markov approximation
        self.initialCondition = initialCondition  # Initial condition for the probability amplitude
        self.cutOff = cutOff  # Cut-off parameter for integration
        self.numPoints = numPoints  # Number of points for Gauss-Legendre quadrature

        if self.spectralDensityType == SpectralDensityType.LORENTZIAN:
            self.omegaA = params.get('omegaA', self.omega0)  # Central frequency of the spectral density
            self.g = params.get('g', 1 * self.omega0)  # Coupling strength
            self.k = params.get('k', 1 * self.omega0)  # Width of the spectral density
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
        Lorentzian spectral density.
        """
        if np.abs(omega - self.omegaA) / self.omega0 < 1e-7:
            spectrum = 2 * self.g ** 2 / (np.pi * self.k)
        else:
            spectrum = (self.g ** 2 / (2 * np.pi)) * self.k / ((omega - self.omegaA) ** 2 + (self.k / 2) ** 2)
        return spectrum

    def normalizedLorentzianSpectrum(self, omega):
        """
        Lorentzian spectral density normalized to decay rate at omega0.
        """
        return self.lorentzianSpectrum(omega) / (2 * np.pi * self.lorentzianSpectrum(self.omega0))

    def metallicSlabSpectrum(self, omega):
        omegaSI = omega * Constants.EV.value
        self.metalSlab.setOmega(omegaSI)
        G_SI = self.metalSlab.calculateNormalizedGreenFunctionReflectedFastIntegration(
            cutOff=self.cutOff, numPoints=self.numPoints
        )

        for i in range(G_SI.shape[0]):
            for j in range(G_SI.shape[1]):
                if np.imag(G_SI[i, j]) < 0:
                    warnings.warn("Imaginary part of green function invalid.")

        # Dipole moment in SI units (C·m)
        dipole_si = self.dipole*Constants.NM.value *Constants.E_CHARGE.value

        # Correct prefactor (SI units)
        prefactor = (4 * Constants.HBAR.value) / (Constants.C_MS.value ** 2) # As we will normaliza we do not use the prefactor to avoid numerical errors
        spectrum =  omegaSI ** 2 * dipole_si.conj().T @ np.imag(G_SI) @ dipole_si

        return spectrum

    def normalizedMetallicSlabSpectrum(self, omega):
        return self.metallicSlabSpectrum(omega) / (2 * np.pi * self.metallicSlabSpectrum(self.omega0))

    def spectralDensity(self, omega):
        if self.spectralDensityType == SpectralDensityType.LORENTZIAN:
            return self.normalizedLorentzianSpectrum(omega)
        elif self.spectralDensityType == SpectralDensityType.METALLIC_SLAB:
            return self.normalizedMetallicSlabSpectrum(omega)

    def computeSpectralDensityArray(self, num_points):
        if self.spectralDensityType == SpectralDensityType.LORENTZIAN:
            omega_min = max(0, self.omegaA - 1.5 * self.k)
            omega_max = self.omegaA + 2 * self.k
        else:
            omega_min = 0.1 * self.omega0
            omega_max = self.omega0 * 5

        omegas = np.linspace(omega_min, omega_max, num=num_points)

        # Parallelize the computation of the spectral density
        spectralDensityArray = Parallel(n_jobs=-1)(delayed(self.spectralDensity)(omega) for omega in omegas)

        return np.array(spectralDensityArray)

    def kernelIntegrand(self, omega, tau, index=None):
        """
        Integrand for the memory kernel.
        """
        if index is None:
            return self.spectralDensity(omega) * np.exp(-1j * (self.omega0 - omega) * tau)

        return self.spectralDensityArray[index] * np.exp(-1j * (self.omega0 - omega) * tau)

    def computeIntegralKernel(self, tau):
        """
        Numerical integration to compute the memory kernel.
        """
        limit = 50
        eps_rel = 1e-3
        # Select a large enough range for the integral
        if self.spectralDensityType == SpectralDensityType.LORENTZIAN:
            omega_min = max(0, self.omegaA - 1.5 * self.k)
            omega_max = self.omegaA + 2 * self.k
        else:
            omega_min = 0.1*self.omega0
            omega_max = self.omega0 * 2  # A value large enough for the metallic slab spectrum

        real_part = spi.quad(lambda x: np.real(self.kernelIntegrand(x, tau)), omega_min, omega_max, limit=limit, epsrel=eps_rel)[0]
        imag_part = spi.quad(lambda x: np.imag(self.kernelIntegrand(x, tau)), omega_min, omega_max, limit=limit, epsrel=eps_rel)[0]
        return real_part + 1j * imag_part

    def computeIntegralKernelFastIntegration(self, tau):
        """
        Numerical integration to compute the memory kernel using Gauss-Legendre quadrature.
        """
        # Select range based on the spectral density type
        if self.spectralDensityType == SpectralDensityType.LORENTZIAN:
            omega_min = max(0, self.omegaA - 1.5 * self.k)
            omega_max = self.omegaA + 2 * self.k
        else:
            omega_min = 0.1 * self.omega0
            omega_max = self.omega0 + self.cutOff/(self.metalSlab.z/Constants.NM.value)

        # Gauss-Legendre quadrature points and weights
        x, w = np.polynomial.legendre.leggauss(self.numPoints)

        # Transform interval [omega_min, omega_max] to [-1, 1]
        mid = (omega_max + omega_min) / 2
        half_width = (omega_max - omega_min) / 2

        # Perform the integration using Gauss-Legendre quadrature
        integral_real = 0.0
        integral_imag = 0.0

        for i in range(self.numPoints):
            omega = mid + half_width * x[i]
            integrand = self.kernelIntegrand(omega, tau, i)
            integral_real += w[i] * np.real(integrand)
            integral_imag += w[i] * np.imag(integrand)

        # Final scaling by the interval width
        integral_real *= half_width
        integral_imag *= half_width

        return integral_real + 1j * integral_imag

    def computeMarkovKernel(self):
        """
        Compute the kernel in the Markov approximation.
        """
        return np.pi * self.spectralDensity(self.omega0)

    def computeKernelArray(self, taus):
        if self.markov:
            kernel_value = self.computeMarkovKernel()
            return np.full_like(taus, kernel_value, dtype=complex)

        # Precompute the kernel on a coarser grid
        taus_coarse = np.linspace(taus[0], taus[-1], len(taus) // 10)
        kernel_coarse = np.array([self.computeIntegralKernelFastIntegration(tau) for tau in taus_coarse])

        # Interpolate
        kernel_interp = interp1d(taus_coarse, kernel_coarse, kind='cubic', fill_value="extrapolate")

        return kernel_interp(taus)

    def convolutionIntegral(self, cE, kValues, dt, n):
        """
        Efficient convolution using trapezoidal integration.
        """
        return fftconvolve(cE[:n], kValues[:n], mode='valid')[0] * dt

    def computeProbability(self, T=10.0, dt=0.01):
        N = int(T / dt)
        times = np.linspace(0, T, N)
        cE = np.zeros(N, dtype=complex)
        cE[0] = self.initialCondition
        self.spectralDensityArray = self.computeSpectralDensityArray(num_points=self.numPoints)

        if self.markov:
            # Markov approximation
            gamma = 2 * np.pi * self.spectralDensity(self.omega0)
            cE = np.exp(-gamma * times)
            return times, np.abs(cE) ** 2
        else:
            # Precompute kernel values
            kernels = self.computeKernelArray(times)

            # RK4 implementation similar to working script
            for n in range(1, N):
                def f(c):
                    return -self.convolutionIntegral(cE, kernels, dt, n)

                k1 = dt * f(cE[n - 1])
                k2 = dt * f(cE[n - 1] + k1 / 2)
                k3 = dt * f(cE[n - 1] + k2 / 2)
                k4 = dt * f(cE[n - 1] + k3)

                cE[n] = cE[n - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

            return times, np.abs(cE) ** 2




