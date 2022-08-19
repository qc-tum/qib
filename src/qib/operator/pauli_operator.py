import numpy as np
from scipy import sparse
from typing import Sequence
from qib.operator import AbstractOperator


class PauliString:
    """
    Pauli string (sequence of identity and Pauli matrices X, Y, Z),
    together with a global phase factor.

    Using check matrix representation (similar to Qiskit convention)
    by storing binary arrays `z` and `x`. The logical Pauli string is
    .. math:: (-i)^q \otimes_{k=0}^{n-1} (-i)^{z_k x_k} Z^{z_k} X^{x_k}
    """
    def __init__(self, data):
        if isinstance(data, str):
            # string representation, like "iYZXZ"
            raise ValueError("not implemented yet")
        elif isinstance(data, tuple):
            # unpack arguments
            z, x, q = data
            self.z = np.array(z, copy=False, dtype=int)
            self.x = np.array(x, copy=False, dtype=int)
            if self.z.ndim != 1 or self.x.ndim != 1:
                raise ValueError("'z' and 'x' parameters must be one-dimensional arrays")
            if self.z.shape != self.x.shape:
                raise ValueError("dimensions of 'z' and 'x' parameters must agree")
            self.q = int(q) % 4

    @property
    def nqubits(self):
        """
        Number of qubits, i.e., length of Pauli string, including identities.
        """
        return len(self.z)

    def as_matrix(self):
        """
        Generate the sparse matrix representation of the Pauli string.
        """
        X = sparse.csr_matrix([[ 0.,  1.], [ 1.,  0.]])
        Z = sparse.csr_matrix([[ 1.,  0.], [ 0., -1.]])
        op = sparse.identity(1)
        for i in range(self.nqubits):
            op = sparse.kron(op, Z**self.z[i] @ X**self.x[i])
        # only use complex type when necessary
        phase = [1., -1j, -1., 1j][(self.q + np.dot(self.z, self.x)) % 4]
        return phase * op


class WeightedPauliString:
    """
    Pauli string with a weight factor.
    """
    def __init__(self, paulis: PauliString, weight):
        self.paulis = paulis
        self.weight = weight

    @property
    def nqubits(self):
        """
        Number of qubits, i.e., length of Pauli string, including identities.
        """
        return self.paulis.nqubits

    def as_matrix(self):
        """
        Generate the sparse matrix representation of the weighted Pauli string.
        """
        return self.weight * self.paulis.as_matrix()


class PauliOperator(AbstractOperator):
    """
    An operator consisting of Pauli strings.
    """
    def __init__(self, pstrings: Sequence[WeightedPauliString]=[]):
        # consistency check
        nqs = [ps.nqubits for ps in pstrings]
        if len(set(nqs)) > 1:
            raise ValueError("all Pauli strings must have the same length")
        self.pstrings = list(pstrings)

    def as_matrix(self):
        """
        Generate the (sparse) matrix representation of the operator.
        """
        if self.pstrings:
            op = self.pstrings[0].as_matrix()
            for ps in self.pstrings[1:]:
                op += ps.as_matrix()
            return op
        else:
            # dimensions are not specified
            return 0
