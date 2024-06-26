from copy import copy
from typing import Sequence
import numpy as np
from qib.operator import Gate, ControlInstruction, MeasureInstruction
from qib.field import Field
from qib.tensor_network import SymbolicTensor, SymbolicTensorNetwork, TensorNetwork
from qib.util import map_particle_to_wire

class Circuit:
    """
    A quantum circuit consists of a list of quantum gates
    and control instructions (e.g. measure, barrier, delay, etc.).

    We follow the convention that the first gate in the list is applied first.
    """

    def __init__(self, gates: Sequence[Gate | ControlInstruction] = None):
        if gates is None:
            self.gates = []
        else:
            self.gates = list(gates)

    def append_gate(self, gate: Gate | ControlInstruction):
        """
        Append a quantum gate or a control instruction.
        """
        self.gates.append(copy(gate))

    def append_circuit(self, other):
        """
        Append the gates from another quantum circuit to the current circuit.
        """
        for g in other.gates:
            self.gates.append(copy(g))

    def prepend_gate(self, gate: Gate | ControlInstruction):
        """
        Prepend a quantum gate or a control instruction.
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
    
    def particles(self):
        """
        Set of all quantum particles appearing in the circuit.
        """
        wires_set = set()
        for gate in self.gates:
            wires_set.update(gate.particles())
        return sorted(wires_set, key=lambda p: p.index)
            
    def clbits(self):
        """
        Set of all classical bits appearing in the circuit.
        """
        bits_set = set()
        for gate in self.gates:
            if type(gate) is MeasureInstruction:
                bits_set.update(gate.memory())
        return sorted(bits_set)

    def inverse(self):
        """
        Construct the "inverse" circuit: reversed list of adjoint gates.
        """
        return Circuit([g.inverse() for g in reversed(self.gates)])

    def as_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the circuit.
        """
        # if circuit contains control instructions, raise a warning
        self._control_instructions_warning()
        if not self.gates:
            raise RuntimeError(
                "missing gates, hence cannot compute matrix representation of circuit")
        
        isFirst = True
        for gate in self.gates:
            if isinstance(gate, ControlInstruction): continue
            if isFirst:
                mat = gate.as_circuit_matrix(fields)
                isFirst = False
            else:
                mat = gate.as_circuit_matrix(fields) @ mat
        
        return mat

    def as_tensornet(self):
        """
        Generate a tensor network representation of the circuit.
        """
        # if circuit contains control instructions, raise a warning
        self._control_instructions_warning()
        # create a tensor network consisting of identity wires
        stn = SymbolicTensorNetwork()
        # virtual tensor for open axes
        wiredims = []
        fields = self.fields()
        for f in fields:
            wiredims += f.lattice.nsites * [f.local_dim]
        # total number of wires
        nwires = len(wiredims)
        # connect input and output open axes
        stn.add_tensor(SymbolicTensor(-1, 2*wiredims,
                       2*list(range(nwires)), None))
        stn.generate_bonds()
        net = TensorNetwork(stn, {})
        assert net.is_consistent()
        for gate in self.gates:
            if isinstance(gate, ControlInstruction): continue
            prtcl = gate.particles()
            iwire = [map_particle_to_wire(fields, p) for p in prtcl]
            if any(iw < 0 for iw in iwire):
                raise RuntimeError("particle not found among fields")
            gate_net = gate.as_tensornet()
            assert gate_net.num_open_axes == 2*len(prtcl)
            net.merge(gate_net, list(
                zip(iwire, list(range(len(prtcl), 2*len(prtcl))))))
            # permute open axes
            perm = list(range(2*nwires))
            for i in iwire:
                perm.remove(i)
            perm += iwire
            net.transpose(np.argsort(perm))
            assert net.is_consistent()
        return net

    def as_qasm(self):
        """
        Generate a list of Qobj OpenQASM instructions representation of the circuit.
        """
        instructions = []
        for gate in self.gates:
            instructions.append(gate.as_qasm())
        return instructions
    
    def _control_instructions_warning(self):
        """
        Check if the circuit contains control instructions.
        If it does, raise a warning.
        """
        for gate in self.gates:
            if isinstance(gate, ControlInstruction):
                print("WARNING: Circuit contains control instructions.\
                      They will be omitted in the conversion.")
                return True
        return False