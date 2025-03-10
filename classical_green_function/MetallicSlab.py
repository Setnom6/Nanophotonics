from enum import Enum
import numpy as np
from typing import List, Any
import scipy.integrate as spi


class Constants(Enum):
    C_MS = 299792458 # c in m/s
    EV = 1.5192678e15 # eV to s^-1
    NM = 1e-9 # nm to m
    EPSILON_0 = 8.85e-12


class Metals(Enum):
    GOLD = (9.0, 0.07, 9.5)
    SILVER = (9.2, 0.02, 4.0)
    COPPER = (10.8, 0.08, 10.0)
    ALUMINUM = (15.0, 0.15, 1.0)
    TUNGSTEN = (8.8, 0.35, 15.0)

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

        return (kParallel * expFactor / kz1) * (termS + termP)

    def _integrateMatrix(self, func, a, b, limit):
        result = np.zeros((3, 3), dtype=complex)
        for i in range(3):
            for j in range(3):
                real_part = spi.quad(lambda x: np.real(func(x)[i, j]), a, b, limit=limit, epsabs=1e-6)[0]
                imag_part = spi.quad(lambda x: np.imag(func(x)[i, j]), a, b, limit=limit, epsabs=1e-6)[0]
                result[i, j] = real_part + 1j * imag_part
        return result

    def calculateGreenFunctionReflected(self, cutOff=10, limit=1000):
        k = self.omega/Constants.C_MS.value
        kSingularity = k
        delta = kSingularity*1e-6

        integral1 = self._integrateMatrix(self._integrand, 0, kSingularity-delta, limit)
        integral2 = self._integrateMatrix(self._integrand,kSingularity+delta , cutOff/self.z, limit)

        return integral1 + integral2

    def calculateImaginaryGreenFunctionHomogeneusSpace(self, epsilon):
        k1 = self._calculateKzi(0, epsilon)
        return (2 / 3) * k1 ** 3 # This comes from the limit computed by hand

    def calculateNormalizedGreenFunctionReflected(self, cutOff=10, limit=1000):
        GReflected = self.calculateGreenFunctionReflected(cutOff)
        GHomogeneusImaginary = self.calculateImaginaryGreenFunctionHomogeneusSpace(self.epsilonList[1])

        return GReflected / GHomogeneusImaginary