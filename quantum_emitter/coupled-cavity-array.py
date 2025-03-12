import numpy as np
import matplotlib.pyplot as plt


def build_HB(N, omega_a=1.0, J=0.2):
    """
    Builds the Hamiltonian matrix H_B with periodic boundary conditions.
    """
    H = np.zeros((N, N), dtype=complex)

    # Diagonal terms
    np.fill_diagonal(H, omega_a)

    # Nearest-neighbor couplings
    for i in range(N - 1):
        H[i, i + 1] = -J
        H[i + 1, i] = -J

    # Periodic boundary condition
    H[0, N - 1] = -J
    H[N - 1, 0] = -J

    return H


def build_HB2(N, omega_a=1.0, J=0.2, J2=0.05):
    """
    Builds the Hamiltonian matrix H_B2 with second-order nearest-neighbor couplings.
    """
    H = build_HB(N, omega_a, J)

    # Second-nearest-neighbor couplings
    for i in range(N - 2):
        H[i, i + 2] = -J2
        H[i + 2, i] = -J2

    # Periodic boundary conditions for second-nearest neighbors
    H[0, N - 2] = -J2
    H[N - 2, 0] = -J2
    H[1, N - 1] = -J2
    H[N - 1, 1] = -J2

    return H


def get_omega_k(H):
    """
    Computes the eigenvalues of H directly in position space.
    """
    eigenvalues = np.linalg.eigvalsh(H)  # Only eigenvalues are needed
    return np.sort(eigenvalues)  # Ensure proper ordering


def get_omega_k_theoretical(N, omega_a=1.0, J=0.2, J2=0.05):
    """
    Computes the theoretical dispersion relation ω(k) and finds the index mapping.
    """
    k_vals = 2 * np.pi * (np.arange(N)-N/2) / N  # Momentum space values

    # Theoretical dispersion relations
    omega_k1 = omega_a - 2 * J * np.cos(k_vals)
    omega_k2 = omega_a - 2 * J * np.cos(k_vals) - 2 * J2 * np.cos(2 * k_vals)

    # Sorted version for ordering reference
    omega_k1_sorted = np.sort(omega_k1)

    # Generate mapping of indices from sorted to original
    good_ordering = []
    used_indices = set()

    for val in omega_k1:
        index = np.where(omega_k1_sorted == val)[0]  # Get all matching indices
        for idx in index:
            if idx not in used_indices:
                good_ordering.append(idx)
                used_indices.add(idx)
                break  # Ensure each value is used only once

    return k_vals, omega_k1, omega_k2, good_ordering


def sort_array_given_order(good_ordering, array):
    """
    Reorders the given array according to the order in good_ordering.
    """
    return np.array([array[i] for i in good_ordering])


def plot_all_dispersion(N, omega_a=1.0, J=0.2, J2=0.05):
    """
    Plots numerical and theoretical dispersions for both first-order (H_B) and second-order (H_B2) Hamiltonians.
    """
    # Build Hamiltonians and compute eigenvalues
    H_B = build_HB(N, omega_a, J)
    H_B2 = build_HB2(N, omega_a, J, J2)

    # Compute theoretical dispersion relations
    k_vals, omega_k_theoretical_B, omega_k_theoretical_B2, good_ordering = get_omega_k_theoretical(N, omega_a, J, J2)

    # k-ordered numerical results
    omega_k_numerical_B = sort_array_given_order(good_ordering, get_omega_k(H_B))
    omega_k_numerical_B2 = sort_array_given_order(good_ordering, get_omega_k(H_B2))


    # Plot everything
    plt.figure(figsize=(8, 5))
    plt.plot(k_vals, omega_k_theoretical_B, 'r--', label=r'Theoretical $\omega(k)$ (1st order)')
    plt.scatter(k_vals, omega_k_numerical_B, color='b', label=r'Numerical $\omega(k)$ (1st order)', zorder=3)

    plt.plot(k_vals, omega_k_theoretical_B2, 'g--', label=r'Theoretical $\omega(k)$ (2nd order)')
    plt.scatter(k_vals, omega_k_numerical_B2, color='m', label=r'Numerical $\omega(k)$ (2nd order)', zorder=3)

    plt.xlabel(r'Wavevector $k$')
    plt.ylabel('Eigenenergy $\omega(k)$')
    plt.title(f'Dispersion Relation Comparison (N={N})')
    plt.legend()
    plt.grid()
    plt.show()


# Run the function
N = 20
J = 0.2
J2 = 0.05
omega_a = 1.0
plot_all_dispersion(N, omega_a, J, J2)





