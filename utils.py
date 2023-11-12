import pennylane as qml
from pennylane import numpy as np

def uniform_superposition(wires):
    for wire in wires:
        qml.Hadamard(wires=wire)

def grover_oracle(omegas, wires):
    for omega in omegas:
        qml.FlipSign(omega, wires=wires)

def tensor_to_qubits(tensor, sizes):
    """
    This function is used for tensor rank decomposition.
    It encodes a 3-tensor into qubits.
    """
    
    # Get the sizes
    _, size, n_qubits = sizes
    assert tensor.shape == (size, size, size)
    
    # Concatenate bit strings for each value in the tensor
    bit_str = ""
    for j in range(size):
        for k in range(size):
            for l in range(size):
                bit_str += f"{tensor[j, k, l] % (2**n_qubits):b}".zfill(n_qubits)

    return [int(i) for i in bit_str]