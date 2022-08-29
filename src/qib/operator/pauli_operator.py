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
    def __init__(self, z, x, q):
        self.z = np.array(z, copy=False, dtype=int)
        self.x = np.array(x, copy=False, dtype=int)
        if self.z.ndim != 1 or self.x.ndim != 1:
            raise ValueError("'z' and 'x' parameters must be one-dimensional arrays")
        if self.z.shape != self.x.shape:
            raise ValueError("dimensions of 'z' and 'x' parameters must agree")
        if not set(self.z).issubset({ 0, 1 }):
            raise ValueError("only allowed entries in 'z' are 0 or 1")
        if not set(self.x).issubset({ 0, 1 }):
            raise ValueError("only allowed entries in 'x' are 0 or 1")
        self.q = int(q) % 4

    @classmethod
    def identity(cls, nqubits: int):
        """
        Construct the Pauli string representation of the identity operation.
        """
        z = np.zeros(nqubits, dtype=int)
        x = np.zeros(nqubits, dtype=int)
        return cls(z, x, 0)

    @classmethod
    def from_string(cls, nqubits: int, s: str):
        """
        Construct a Pauli string from a literal string representation, like "iYZXZ".
        """
        raise NotImplementedError

    @classmethod
    def from_single_paulis(cls, nqubits: int, *args, **kwargs):
        """
        Construct a Pauli string from a list of single Pauli matrices, like
        ('Y', 1), ('X', 4), ('Z', 5).
        """
        z = np.zeros(nqubits, dtype=int)
        x = np.zeros(nqubits, dtype=int)
        for arg in args:
            # unpack
            s, i = arg
            if i < 0 or i >= nqubits:
                raise ValueError(f"index {i} out of range, number of qubits is {nqubits}")
            # overwrite entries in case same index appears multiple times
            if s == 'I':
                z[i] = 0
                x[i] = 0
            elif s == 'X':
                z[i] = 0
                x[i] = 1
            elif s == 'Y':
                z[i] = 1
                x[i] = 1
            elif s == 'Z':
                z[i] = 1
                x[i] = 0
            else:
                raise ValueError(f"invalid Pauli matrix denominator, received {s}, expecting 'I', 'X', 'Y' or 'Z'")
        q = 0
        for key, value in kwargs.items():
            if key == "q":
                q = value
        return cls(z, x, q)

    @property
    def nqubits(self):
        """
        Number of qubits, i.e., length of Pauli string, including identities.
        """
        return len(self.z)

    def get_pauli(self, i: int):
        """
        Get the Pauli matrix at index `i` as string 'I', 'X', 'Y' or 'Z'.
        """
        if self.z[i] == 0:
            if self.x[i] == 0:
                return 'I'
            else:
                return 'X'
        else:
            if self.x[i] == 0:
                return 'Z'
            else:
                return 'Y'

    def set_pauli(self, s: str, i: int):
        """
        Set an individual Pauli matrix at index `i`.
        """
        if s == 'I':
            self.z[i] = 0
            self.x[i] = 0
        elif s == 'X':
            self.z[i] = 0
            self.x[i] = 1
        elif s == 'Y':
            self.z[i] = 1
            self.x[i] = 1
        elif s == 'Z':
            self.z[i] = 1
            self.x[i] = 0
        else:
            raise ValueError(f"invalid Pauli matrix denominator, received {s}, expecting 'I', 'X', 'Y' or 'Z'")

    def __matmul__(self, other):
        """
        Logical matrix multiplication of two Pauli strings.
        """
        z_prod = np.mod(self.z + other.z, 2)
        x_prod = np.mod(self.x + other.x, 2)
        q_prod = int(  np.dot(self.z, self.x)
                     + np.dot(other.z, other.x)
                     - np.dot(z_prod, x_prod)
                     + 2*np.dot(self.x, other.z))   # sign factor from flipping X <-> Z
        return PauliString(z_prod, x_prod, self.q + other.q + q_prod)

    def as_matrix(self):
        """
        Generate the sparse matrix representation of the Pauli string.
        """
        X = sparse.csr_matrix([[ 0.,  1.], [ 1.,  0.]])
        Z = sparse.csr_matrix([[ 1.,  0.], [ 0., -1.]])
        op = sparse.identity(1)
        for i in range(self.nqubits):
            op = sparse.csr_matrix(sparse.kron(op, Z**self.z[i] @ X**self.x[i]))
            op.eliminate_zeros()
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

    def add_pauli_string(self, ps: WeightedPauliString):
        """
        Add a weighted Pauli string.
        """
        self.pstrings.append(ps)

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
