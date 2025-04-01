import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

# Functions

def lorentzian_spectrum_normalized(g, k, omega_a, omega_0, omega):
    decay_rate_at_omega_0 = g**2 / (2 * np.pi) * k / ((omega_0 - omega_a)**2 + (k / 2)**2)
    return (g**2 / (2 * np.pi) * k / ((omega - omega_a)**2 + (k / 2)**2)) / decay_rate_at_omega_0

def kernel_integrand(omega, g, k, omega_a, omega_0, tau):
    return lorentzian_spectrum_normalized(g, k, omega_a, omega_0, omega) * np.exp(-1j * (omega_0 - omega) * tau)

def compute_integral_kernel(g, k, omega_a, omega_0, tau):
    real_part = spi.quad(lambda x: np.real(kernel_integrand(x, g, k, omega_a, omega_0, tau)),
                         0, omega_a + 5 * k, limit=100)[0]
    imag_part = spi.quad(lambda x: np.imag(kernel_integrand(x, g, k, omega_a, omega_0, tau)),
                         0, omega_a + 5 * k, limit=100)[0]
    return real_part + 1j * imag_part

def analytical_kernel(g, k, omega_a, omega_0, tau):
    delta = omega_a - omega_0
    decay_rate = g**2 / (2 * np.pi) * k / ((omega_0 - omega_a)**2 + (k / 2)**2)
    return g**2 * np.exp(1j * delta * tau) * np.exp(-k * tau / 2) / decay_rate

def convolution_integral(c_e, k_values, dt, n):
    integral = 0
    for m in range(n):
        integral += (k_values[n-m] * c_e[m] + k_values[n-m-1] * c_e[m+1]) * dt / 2  # Trapezoidal rule
    return integral

# Solve the integro-differential equation and compute probability P(t)
def compute_probability(g, k, omega_a, omega_0, T=10.0, dt=0.01, method="integral"):
    N = int(T / dt)  # Number of time steps
    taus = np.linspace(0, T, N)

    # Select kernel method
    if method == "integral":
        kernels = np.array([compute_integral_kernel(g, k, omega_a, omega_0, tau) for tau in taus])
    elif method == "analytical":
        kernels = np.array([analytical_kernel(g, k, omega_a, omega_0, tau) for tau in taus])
    else:
        raise ValueError("Invalid method. Choose 'integral' or 'analytical'.")

    # Initialize c_e(t)
    c_e = np.zeros(N, dtype=complex)
    c_e[0] = 1.0  # Initial condition: c_e(0) = 1

    # Runge-Kutta 4 method
    for n in range(1, N):
        t_n = n * dt

        # Define the function f(t, c) = -∫ k(t-s) c(s) ds
        def f(t, c):
            return -convolution_integral(c_e, kernels, dt, n)

        # RK4 steps
        k1 = dt * f(t_n, c_e[n-1])
        k2 = dt * f(t_n + dt/2, c_e[n-1] + k1/2)
        k3 = dt * f(t_n + dt/2, c_e[n-1] + k2/2)
        k4 = dt * f(t_n + dt, c_e[n-1] + k3)

        c_e[n] = c_e[n-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

    # Compute probability P(t) = |c_e(t)|^2
    P_t = np.abs(c_e)**2
    return taus, P_t

# Parameters
g = 1
k = 1
delta = 0
omega_a = 10
omega_0 = omega_a - delta

# Compute probability using both methods
taus_integral, P_t_integral = compute_probability(g, k, omega_a, omega_0, method="integral")
taus_analytical, P_t_analytical = compute_probability(g, k, omega_a, omega_0, method="analytical")

# Plot Probability P(t) for both methods
plt.figure(figsize=(10, 6))
plt.plot(taus_integral, P_t_integral, label='Numerical Integral Kernel', color='b', linestyle='-')
plt.plot(taus_analytical, P_t_analytical, label='Analytical Kernel', color='r', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Probability $P(t)$')
plt.title('Comparison of Probability Evolution Over Time')
plt.legend()
plt.grid(True)
plt.show()
