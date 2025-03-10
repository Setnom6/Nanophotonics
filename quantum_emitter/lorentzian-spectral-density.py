import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

# Functions

def lorentzian_spectrum_normalized(g, k, omega_a, omega_0, omega):
    decay_rate_at_omega_0 = g**2 / (2 * np.pi) * k / ((omega_0 - omega_a)**2 + (k / 2)**2)
    return (g**2 / (2 * np.pi) * k / ((omega - omega_a)**2 + (k / 2)**2))/decay_rate_at_omega_0

def kernel_integrand(omega, g, k, omega_a, omega_0, tau):
    return lorentzian_spectrum_normalized(g, k, omega_a, omega_0, omega) * np.exp(1j * (omega_0-omega) * tau)

def compute_integral(g, k, omega_a, omega_0, tau):
    real_part = spi.quad(lambda x: np.real(kernel_integrand(x, g, k, omega_a, omega_0, tau)), 0, omega_a+4*k)[0]
    imag_part = spi.quad(lambda x: np.imag(kernel_integrand(x, g, k, omega_a, omega_0, tau)), 0, omega_a+4*k)[0]
    return real_part + 1j * imag_part

def analytical_kernel(g, k, omega_a, omega_0, tau):
    delta = omega_a - omega_0
    decay_rate = g**2 / (2 * np.pi) * k / ((omega_0 - omega_a)**2 + (k / 2)**2)
    return g**2 * np.exp(-1j * delta * tau) * np.exp(-k * tau / 2) / decay_rate

# Parameters

g = 1
k = 1
delta = 5
omega_a = 10
omega_0 = omega_a - delta
tau_min = 0
tau_max = 10

# Array of taus

taus = np.linspace(tau_min, tau_max, 100)
kernels_integrated = np.array([compute_integral(g, k, omega_a, omega_0, tau) for tau in taus])
kernels_analytical = np.array([analytical_kernel(g, k, omega_a, omega_0, tau) for tau in taus])

# Plot

plt.figure(figsize=(10, 6))
plt.scatter(taus, np.real(kernels_integrated), label='Re Integral', color='blue', marker='o')
plt.scatter(taus, np.imag(kernels_integrated), label='Im Integral', color='red', marker='x')
plt.plot(taus, np.real(kernels_analytical), label='Re Analytical', color='blue', linestyle='--')
plt.plot(taus, np.imag(kernels_analytical), label='Im Analytical', color='red', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Kernel Value')
plt.title('Comparison of Calculated and Analytical Kernels')
plt.legend()
plt.grid(True)
plt.show()





