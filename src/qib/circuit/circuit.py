from typing import Sequence
from copy import copy
from qib.operator import Gate
from qib.field import Field


class Circuit:
    """
    A quantum circuit consists of a list of quantum gates.

    We follow the convention that the first gate in the list is applied first.
    """
    def __init__(self, gates: Sequence[Gate]=[]):
        self.gates = list(gates)

    def append_gate(self, gate: Gate):
        """
        Append a quantum gate.
        """
        self.gates.append(copy(gate))

    def append_circuit(self, other):
        """
        Append the gates from another quantum circuit to the current circuit.
        """
        for g in other.gates:
            self.gates += copy(g)

    def prepend_gate(self, gate: Gate):
        """
        Prepend a quantum gate.
        """
        self.gates.insert(0, copy(gate))

    def fields(self):
        """
        List of all fields appearing in the circuit.
        """
        flist = []
        for g in self.gates:
            for f in g.fields():
                if f not in flist:
                    flist.append(f)
        return flist

    def inverse(self):
        """
        Construct the "inverse" circuit: reversed list of adjoint gates.
        """
        return Circuit([g.inverse() for g in reversed(self.gates)])

    def as_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the circuit.
        """
        if not self.gates or len(self.gates)==0:
            raise RuntimeError("missing gates, hence cannot compute matrix representation of circuit")
        # Warning: do not use the ones saved in self.fields() because the order is not fixed
        if not fields:
            raise RuntimeError("missing fields, hence cannot compute matrix representation of circuit")
        mat = self.gates[0]._circuit_matrix(fields)
        for g in self.gates[1:]:
            mat = g._circuit_matrix(fields) @ mat
        return mat
