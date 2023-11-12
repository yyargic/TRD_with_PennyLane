import pennylane as qml
from pennylane import numpy as np
from arithmetic import compute_tensor
from utils import uniform_superposition, tensor_to_qubits
import matplotlib.pyplot as plt

QUBITS_PER_NUM = 1
PROBLEM_SIZE = 2
SOLUTION_RANK = 3

sizes = (SOLUTION_RANK, PROBLEM_SIZE, QUBITS_PER_NUM)
params_per_edge = SOLUTION_RANK * PROBLEM_SIZE * QUBITS_PER_NUM
wires_a = list(range(0, params_per_edge))
wires_b = list(range(params_per_edge, 2*params_per_edge))
wires_c = list(range(2*params_per_edge, 3*params_per_edge))
wires_counter = list(range(3*params_per_edge, 3*params_per_edge + (PROBLEM_SIZE**3) * QUBITS_PER_NUM))

# The tensor that we wish to decompose
complex_mul_tensor = np.array([[[1, 0], [0, -1]], [[0, 1], [1, 0]]])
assert complex_mul_tensor.shape == (PROBLEM_SIZE, PROBLEM_SIZE, PROBLEM_SIZE)
omega = tensor_to_qubits(complex_mul_tensor, sizes)

dev = qml.device("default.qubit", wires = wires_a + wires_b + wires_c + wires_counter)

@qml.qnode(dev)
def grover_TRD(omega, wires_a, wires_b, wires_c, wires_counter, iterations=1):

    # Bring input to uniform superposition
    uniform_superposition(wires_a + wires_b + wires_c)

    for _ in range(iterations):

        # Compute the tensor from a,b,c
        compute_tensor(wires_a, wires_b, wires_c, wires_counter, sizes)

        # Apply Grover's oracle
        qml.FlipSign(omega, wires=wires_counter)

        # Uncompute the tensor
        qml.adjoint(compute_tensor)(wires_a, wires_b, wires_c, wires_counter, sizes)

        # Apply Grover's diffusion operator
        qml.GroverOperator(wires = wires_a + wires_b + wires_c)

    return qml.probs(wires=wires_a)

result = grover_TRD(omega, wires_a, wires_b, wires_c, wires_counter, iterations=1)
plt.bar(range(2 ** len(wires_a)), result)
plt.xlabel("Basic states")
plt.ylabel("Probability")
plt.show()
