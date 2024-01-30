import abc
from typing import Sequence
from qib.operator import AbstractOperator
from qib.field import Field, Particle, Qubit


class Measurement(AbstractOperator):
    """
    Measurement operator for a quantum circuit.

    A quantum measurement represents the corresponding operator that measures a quantum bit into a classical bit,
    and can store the qubits (or in general quantum particles) it operates on.
    """

    def __init__(self, qubits: Sequence[Qubit] = None, clbits: Sequence[int] = None):
        self._assign_qubits_clbits(qubits, clbits)

    def is_unitary(self):
        """
        A quantum measurement operator is never unitary.
        """
        return False

    def is_hermitian(self):
        """
        A quantum measurement operator is never Hermitian.
        """
        return False

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this operator is performed on.
        """
        if self.qubits:
            return len(self.qubits)
        return 0

    def particles(self):
        """
        Return the list of quantum particles the operator is performed on.
        """
        if self.qubits:
            return [q for q in self.qubits]
        return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the operator is performed on.
        """
        if self.qubits:
            return [q.field for q in self.qubits]
        return []

    def memory(self):
        """
        Return the list of memory slots the operator will store the results in.
        """
        if self.clbits:
            return [c for c in self.clbits]
        return []

    def on(self, qubits: Sequence[Qubit], clbits: Sequence[int] = None):
        """
        Act on the specified qubits.
        """
        self._assign_qubits_clbits(qubits, clbits)

        # enable chaining
        return self

    def as_matrix(self):
        """
        Generate the (sparse) matrix representation of the operator.
        """
        # TODO: To be decided, if needed
        raise NotImplementedError() 

    def as_openQASM(self):
        """
        Generate a Qobj OpenQASM representation of the operator.
        """
        return {
            "name": "measure",
            "qubits": [q.index for q in self.qubits],
            "memory": [c for c in self.clbits]
        }

    def _assign_qubits_clbits(self, qubits, clbits):
        """
        Assign qubits and classical bits
        """
        # check that the number of qubits and classical bits match
        if qubits and clbits:
            if len(qubits) != len(clbits):
                raise ValueError(
                    "Number of qubits and classical bits must match")

        if qubits:
            self.qubits = qubits
            self.clbits = clbits if clbits else [q.index for q in self.qubits]
        else:
            self.qubits = self.clbits = None

    def __copy__(self):
        """
        Create a copy of the operator.
        """
        return Measurement(self.qubits, self.clbits)

    def __eq__(self, other):
        """
        Check if measurement operators are equivalent.
        """
        if type(other) == type(self) and other.qubits == self.qubits and other.clbits == self.clbits:
            return True
        return False
