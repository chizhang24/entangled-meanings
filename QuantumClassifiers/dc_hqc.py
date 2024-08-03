import pennylane as qml
from pennylane import numpy as np
from .divide_and_conquer import Encoding


def _mry(q, weights):
    n = len(q)
    for i in range(n):
        qml.RY(weights[i], wires=q[i])
        if (i+1) % 2 == 0:
            qml.CNOT(wires=[q[i], q[i-1]])

def hierarchical_circuit(n, q, weights):
    layer = 0
    ops_total = 0
    while True:
        ops_count = n // 2**layer

        _mry(q[ : n : n // ops_count], weights[ops_total : ops_total+ops_count])
        
        ops_total += ops_count
        layer += 1
        if (ops_count <= 1):
            break

    return qml.expval(qml.PauliZ(0))



def config(X):
    n = int(np.ceil(np.log2(len(X[0]))))             
    n = 2**int(np.ceil(np.log2(n)))                  
    N = 2**n-1                                       
    w = 2*n - 1                                      
    X = np.c_[X, np.zeros((len(X), 2**n-len(X[0])))] 

    return n, N, w, X

def circuit(weights, state_vector=None, n=4):
    encode = Encoding(state_vector, 'dc_amplitude_encoding', entangle=True)
    q = encode.output_qubits # or it is complement of the output qubits

    return hierarchical_circuit(n , q, weights)