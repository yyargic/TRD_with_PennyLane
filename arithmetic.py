import pennylane as qml
from pennylane import numpy as np

def add_k_fourier(k, wires):
    """Add k to the number in wires in the Fourier basis."""

    for idx, wire in enumerate(wires):
        qml.RZ(k * np.pi / (2**idx), wires=wire)

def addition(wires_a, wires_b, wires_sol):
    """Add the sum of the numbers in wires_a and wires_b to wires_sol."""

    # Prepare solution qubits to counting
    qml.QFT(wires=wires_sol)

    # Add wires_a to the counter
    for idx, wire in enumerate(wires_a):
        qml.ctrl(add_k_fourier, control=wire)(2**(len(wires_a) - idx - 1), wires_sol)

    # Add wires_b to the counter
    for idx, wire in enumerate(wires_b):
        qml.ctrl(add_k_fourier, control=wire)(2**(len(wires_b) - idx - 1), wires_sol)

    # Return to the computational basis
    qml.adjoint(qml.QFT)(wires=wires_sol)

def multiplication(wires_a, wires_b, wires_sol):
    """Add the product of the numbers in wires_a and wires_b to wires_sol."""

    # Prepare solution qubits to counting
    qml.QFT(wires=wires_sol)

    # Add product to the counter
    for idx_a, wire_a in enumerate(wires_a):
        for idx_b, wire_b in enumerate(wires_b):
            coeff = 2 ** (len(wires_a) + len(wires_b) - idx_a - idx_b - 2)
            qml.ctrl(add_k_fourier, control=[wire_a, wire_b])(coeff, wires_sol)

    # Return to the computational basis
    qml.adjoint(qml.QFT)(wires=wires_sol)

def compute_tensor(wires_a, wires_b, wires_c, wires_sol, sizes):
    """
    This function is used for tensor rank decomposition.
    It computes the 3-tensor for which (wires_a, wires_b, wires_c) is a TRD,
    and adds the result to the registers wires_sol.
    """

    # Get the sizes
    rank, size, n_qubits = sizes
    assert len(wires_a) == len(wires_b) == len(wires_c) == rank * size * n_qubits
    assert len(wires_sol) == (size**3) * n_qubits

    # Prepare solution qubits to counting
    for j in range(size):
        for k in range(size):
            for l in range(size):
                loc_sol = (j * size**2 + k * size + l) * n_qubits
                qml.QFT(wires=wires_sol[loc_sol: loc_sol + n_qubits])

    # Add the product to the counter
    for r in range(rank):
        for j in range(size):
            for k in range(size):
                for l in range(size):
                    loc_0 = r * size * n_qubits
                    loc_a, loc_b, loc_c = loc_0 + j * n_qubits, loc_0 + k * n_qubits, loc_0 + l * n_qubits
                    loc_sol = (j * size**2 + k * size + l) * n_qubits
                    subset_wires_a = wires_a[loc_a: loc_a + n_qubits]
                    subset_wires_b = wires_b[loc_b: loc_b + n_qubits]
                    subset_wires_c = wires_c[loc_c: loc_c + n_qubits]
                    subset_wires_sol = wires_sol[loc_sol: loc_sol + n_qubits]
                    for idx_a, wire_a in enumerate(subset_wires_a):
                        for idx_b, wire_b in enumerate(subset_wires_b):
                            for idx_c, wire_c in enumerate(subset_wires_c):
                                coeff = 2 ** (len(wires_a) + len(wires_b) + len(wires_c) - idx_a - idx_b - idx_c - 3)
                                qml.ctrl(add_k_fourier, control=[wire_a, wire_b, wire_c])(coeff, subset_wires_sol)
                                
    # Return to the computational basis
    for j in range(size):
        for k in range(size):
            for l in range(size):
                loc_sol = (j * size**2 + k * size + l) * n_qubits
                qml.adjoint(qml.QFT)(wires=wires_sol[loc_sol: loc_sol + n_qubits])
