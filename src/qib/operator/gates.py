import abc
import numpy as np
import scipy.sparse.linalg as spla
from qib.operator import AbstractOperator, FieldOperator


class Gate(AbstractOperator):
    """
    Parent class for quantum gates.

    A quantum gate represents the corresponding unitary operation,
    but does not store the qubits (or in general quantum particles) it acts on.
    """

    def is_unitary(self):
        """
        A quantum gate is unitary by definition.
        """
        return True

    @property
    @abc.abstractmethod
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        pass


class PauliXGate(Gate):
    """
    Pauli-X gate.
    """
    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return True

    @property
    def num_wires(self):
        return 1

    def as_matrix(self):
        """
        Generate the matrix representation of the Pauli-X gate.
        """
        return np.array([[ 0.,  1.], [ 1.,  0.]])


class PauliYGate(Gate):
    """
    Pauli-Y gate.
    """
    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return True

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return 1

    def as_matrix(self):
        """
        Generate the matrix representation of the Pauli-Y gate.
        """
        return np.array([[ 0., -1j], [ 1j,  0.]])


class PauliZGate(Gate):
    """
    Pauli-Z gate.
    """
    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return True

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return 1

    def as_matrix(self):
        """
        Generate the matrix representation of the Pauli-Z gate.
        """
        return np.array([[ 1.,  0.], [ 0., -1.]])


class HadamardGate(Gate):
    """
    Hadamard gate.
    """
    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return True

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return 1

    def as_matrix(self):
        """
        Generate the matrix representation of the Hadamard gate.
        """
        return np.array([[ 1.,  1.], [ 1., -1.]]) / np.sqrt(2)


class TimeEvolutionGate(Gate):
    """
    Quantum time evolution gate, i.e., matrix exponential,
    given a field operator (Hamiltonian) `h`:
    .. math:: e^{-i h t}
    """
    def __init__(self, h: FieldOperator, t: float):
        self.h = h
        self.t = t

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        # might be Hermitian in special cases, but difficult to check,
        # so returning False here for simplicity
        return False

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        fields = self.h.fields()
        return sum(f.lattice.nsites for f in fields)

    def as_matrix(self):
        """
        Generate the matrix representation of the time evolution gate.
        """
        hmat = self.h.as_matrix()
        # TODO: exploit that `hmat` is Hermitian for computing matrix exponential
        return spla.expm(-1j*self.t * hmat)
