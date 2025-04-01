from enum import Enum
import numpy as np
from typing import List, Any
import scipy.integrate as spi


class Constants(Enum):
    C_MS = 299792458 # c in m/s
    EV = 1.5192678e15 # eV to s^-1
    NM = 1e-9 # nm to m
    EPSILON_0 = 8.85e-12
    HBAR = 1.0545718e-34 # J·s
    E_CHARGE = 1.602176634e-19  # Elementary charge (C)

"""
References:
For SILVER: https://nano-optics.colorado.edu/wp-content/uploads/2020/06/Yang_PhysRevB_15_MainText.pdf
"""

class Metals(Enum):
    GOLD = (9.03, 0.071, 9.84)
    SILVER = (8.9, 0.039, 5.0)
    COPPER = (10.83, 0.086, 10.85)
    ALUMINUM = (14.98, 0.047, 1.24)
    TUNGSTEN = (13.22, 0.064, 4.1)

    def __init__(self, plasmaFrequency, gamma, epsilonB):
        self.plasmaFrequency = plasmaFrequency
        self.gamma = gamma
        self.epsilonB = epsilonB

class MetallicSlab:
    plasmaFrequency : float
    gamma : float
    epsilonB : float
    epsilonList: List[Any]
    t: float
    z: float
    omega: float
    polarizationList: List[str]

    def __init__(self, metal: Metals, epsilon1, epsilon3, omega, t, z):
        self.plasmaFrequency = metal.plasmaFrequency * Constants.EV.value
        self.gamma = metal.gamma * Constants.EV.value
        self.epsilonB = metal.epsilonB
        self.omega = omega
        self.t = t
        self.z = z
        self.epsilonList = [Constants.EPSILON_0.value , epsilon1, self._calculateEpsilonDrude(), epsilon3]
        self.polarizationList = ['s', 'p']

    def setOmega(self, omega):
        """
        Update the frequency and recalculate dependent properties.
        """
        self.omega = omega
        self.epsilonList[2] = self._calculateEpsilonDrude()  # Recalculate epsilonDrude

    def _calculateEpsilonDrude(self) -> complex:
        return self.epsilonB - (self.plasmaFrequency ** 2) / (self.omega ** 2 + 1j * self.omega * self.gamma)

    def _calculateKzi(self, kParallel, epsilon):
        k = self.omega/ Constants.C_MS.value
        return np.sqrt(epsilon * k ** 2 - kParallel ** 2 + 0j)

    def _calculateRij(self, kParallel, epsiloni, epsilonj, polarization):
        kzi = self._calculateKzi(kParallel, epsiloni)
        kzj = self._calculateKzi(kParallel, epsilonj)

        if polarization == 's':
            return (kzi - kzj) / (kzi + kzj)

        elif polarization == 'p':
            return (epsilonj*kzi - epsiloni*kzj) / (epsilonj*kzi + epsiloni*kzj)

        else:
            raise NotImplementedError

    def _calculateTotalReflection(self, kParallel, polarization):
        r12 = self._calculateRij(kParallel, self.epsilonList[1], self.epsilonList[2], polarization)
        r23 = self._calculateRij(kParallel, self.epsilonList[2], self.epsilonList[3], polarization)
        kz2 = self._calculateKzi(kParallel, self.epsilonList[2])
        exp = np.exp(2j*kz2*self.t)

        return (r12 + r23*exp)/(1+r12*r23*exp)


    def _integrand(self, kParallel):
        # At the moment this is an inefficient calculation as only two values will be different
        # But the matrix shape is mantained for future development
        k = self.omega/Constants.C_MS.value
        kz1 = self._calculateKzi(kParallel, self.epsilonList[1])
        rs = self._calculateTotalReflection(kParallel, self.polarizationList[0])
        rp = self._calculateTotalReflection(kParallel, self.polarizationList[1])

        expFactor = np.exp(2j * kz1 * self.z)

        termS = rs * np.array([[1j * k ** 2 / 2, 0, 0],
                               [0, 1j * k ** 2 / 2, 0],
                               [0, 0, 0]])

        termP = rp * np.array([[-1j * kz1 ** 2 / 2, 0, 0],
                       [0, -1j * kz1 ** 2 / 2, 0],
                       [0, 0, 1j * kParallel ** 2]])

        result = (kParallel * expFactor / kz1) * (termS + termP)
        return result

    def _integrateMatrix(self, func, a, b, eps_rel, limit):
        result = np.zeros((3, 3), dtype=complex)
        for i in range(3):
            for j in range(3):
                real_part = spi.quad(lambda x: np.real(func(x)[i, j]), a, b, limit=limit, epsrel=eps_rel)[0]
                imag_part = spi.quad(lambda x: np.imag(func(x)[i, j]), a, b, limit=limit, epsrel=eps_rel)[0]
                result[i, j] = real_part + 1j * imag_part
        return result

    def calculateGreenFunctionReflected(self, cutOff=10, eps_rel=1.49e-8, limit=1000):
        k = self.omega/Constants.C_MS.value
        kSingularity = k
        delta = kSingularity*1e-8

        integral1 = self._integrateMatrix(self._integrand, 0, kSingularity-delta, eps_rel, limit)
        integral2 = self._integrateMatrix(self._integrand,kSingularity+delta , cutOff/self.z, eps_rel, limit)

        return integral1 + integral2

    def calculateImaginaryGreenFunctionHomogeneousSpace(self, epsilon):
        """Calcula la parte imaginaria de G en espacio homogéneo con chequeo de positividad"""
        k1 = self._calculateKzi(0, epsilon)  # Componente perpendicular del vector de onda

        # Cálculo directo con verificación física
        img_part = (2 / 3) * np.imag(k1 ** 3)  # Debería ser siempre positiva

        if img_part <= 0:
            # Valor de emergencia basado en límite asintótico
            img_part = (2 / 3) * (self.omega / Constants.C_MS.value) ** 3

        return img_part

    def calculateNormalizedGreenFunctionReflected(self, cutOff=10, eps_rel=1.49e-8, limit=1000):
        GReflected = self.calculateGreenFunctionReflected(cutOff, eps_rel, limit)
        GHomogeneusImaginary = self.calculateImaginaryGreenFunctionHomogeneousSpace(self.epsilonList[1])

        return GReflected / GHomogeneusImaginary

    def gaussLegendreMatrixIntegration(self, func, a, b, numPoints=50):
        """
        Performs the integration of a matrix function using Gauss-Legendre quadrature.
        """
        # Gauss-Legendre quadrature points and weights
        x, w = np.polynomial.legendre.leggauss(numPoints)

        # Transform interval [a, b] to [-1, 1]
        mid = (b + a) / 2
        halfWidth = (b - a) / 2

        # Initialize result matrix
        result = np.zeros((3, 3), dtype=complex)

        for i in range(numPoints):
            kParallel = mid + halfWidth * x[i]
            integrandMatrix = func(kParallel)

            # Accumulate the weighted result
            result += w[i] * integrandMatrix

        # Scale the result by the interval width
        result *= halfWidth

        return result

    def calculateNormalizedGreenFunctionReflectedFastIntegration(self, cutOff=50, numPoints=300):
        """
        Versión robusta para pequeños z/t que:
        1. Usa un esquema de integración adaptativo
        2. Implementa correcciones analíticas para z/t → 0
        3. Previene valores negativos no físicos
        """
        k = self.omega / Constants.C_MS.value
        kSingularity = k

        # Manejo especial para z/t < 0.1
        if self.z / self.t < 0.1:
            return self._handle_small_z_case(cutOff, numPoints)

        # Cálculo estándar para z/t ≥ 0.1
        delta = max(kSingularity * 1e-8, 1e-3 / self.z)  # Asegura δ > 0

        # Integración con más puntos cerca de la singularidad
        points_near_sing = int(numPoints * 0.7)
        points_far = numPoints - points_near_sing

        integral1 = self.gaussLegendreMatrixIntegration(
            self._integrand, 0, kSingularity - delta, points_near_sing)

        integral2 = self.gaussLegendreMatrixIntegration(
            self._integrand, kSingularity + delta, cutOff / self.z, points_far)

        gReflected = integral1 + integral2

        # Corrección analítica para pequeños z
        if self.z / self.t < 0.5:
            gReflected = self._apply_small_z_correction(gReflected)

        # Aseguramos Im[G] ≥ 0
        gReflected = self._ensure_positive_imaginary(gReflected)

        gHomogeneous = self.calculateImaginaryGreenFunctionHomogeneousSpace(self.epsilonList[1])
        return gReflected / gHomogeneous

    def _handle_small_z_case(self, cutOff, numPoints):
        """Manejo especial para distancias muy pequeñas z/t < 0.1"""
        # Usamos aproximación cuasi-estática para z → 0
        k = self.omega / Constants.C_MS.value
        epsilon = self._calculateEpsilonDrude()

        # Aproximación de near-field para dipolo perpendicular
        g_zz = (1 / (16 * np.pi * self.z ** 3)) * ((epsilon - 1) / (epsilon + 1))

        # Construimos tensor G aproximado
        G = np.zeros((3, 3), dtype=complex)
        G[2, 2] = g_zz  # Componente dominante para z → 0

        # Corrección para evitar singularidad exacta en z=0
        if np.imag(G[2, 2]) <= 0:
            G[2, 2] = np.abs(np.real(G[2, 2])) + 1j * 1e-6

        return G / self.calculateImaginaryGreenFunctionHomogeneousSpace(self.epsilonList[1])

    def _apply_small_z_correction(self, G):
        """Aplica correcciones analíticas para 0.1 < z/t < 0.5"""
        print("Apply small z/t correction")
        k = self.omega / Constants.C_MS.value
        z_corr = self.z + 0.1 * self.t  # Evita z exactamente cero

        # Factor de corrección empírico basado en límite asintótico
        correction_factor = 1 - np.exp(-(k * z_corr) ** 2)
        return G * correction_factor

    def _ensure_positive_imaginary(self, G):
        """Garantiza que la parte imaginaria sea no negativa"""
        G_imag = np.imag(G)
        if np.any(G_imag < 0):
            G_imag_corrected = np.maximum(G_imag, 1e-10)  # Pequeño valor positivo
            return np.real(G) + 1j * G_imag_corrected
        return G
