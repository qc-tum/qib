import numpy as np
from typing import Sequence
from copy import copy
from qib.operator import Gate
from qib.field import Field
from qib.tensor_network import SymbolicTensor, SymbolicTensorNetwork, TensorNetwork
from qib.util import map_particle_to_wire


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

    def prepend_circuit(self, other):
        """
        Prepend the gates from another quantum circuit to the current circuit.
        """
        self.gates = [copy(g) for g in other.gates] + self.gates

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
        if not self.gates:
            raise RuntimeError("missing gates, hence cannot compute matrix representation of circuit")
        mat = self.gates[0]._circuit_matrix(fields)
        for g in self.gates[1:]:
            mat = g._circuit_matrix(fields) @ mat
        return mat

    def as_tensornet(self, fields: Sequence[Field]):
        """
        Generate a tensor network representation of the circuit.
        """
        # create a tensor network consisting of identity wires
        stn = SymbolicTensorNetwork()
        # virtual tensor for open axes
        wiredims = []
        for f in fields:
            wiredims += f.lattice.nsites * [f.local_dim]
        # total number of wires
        nwires = len(wiredims)
        # connect input and output open axes
        stn.add_tensor(SymbolicTensor(-1, 2*wiredims, 2*list(range(nwires)), None))
        stn.generate_bonds()
        net = TensorNetwork(stn, {})
        assert net.is_consistent()
        for gate in self.gates:
            prtcl = gate.particles()
            iwire = [map_particle_to_wire(fields, p) for p in prtcl]
            if any([iw < 0 for iw in iwire]):
                raise RuntimeError("particle not found among fields")
            gate_net = gate.as_tensornet()
            assert gate_net.num_open_axes == 2*len(prtcl)
            net.merge(gate_net, list(zip(iwire, list(range(len(prtcl), 2*len(prtcl))))))
            # permute open axes
            perm = list(range(2*nwires))
            for i in iwire:
                perm.remove(i)
            perm += iwire
            net.transpose(np.argsort(perm))
            assert net.is_consistent()
        return net
