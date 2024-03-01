import numpy as np
from scipy.linalg import expm
from typing import Sequence
from qib.field import Qubit
from qib.operator import ControlledGate, PauliXGate, RzGate, PhaseFactorGate
from qib.circuit import Circuit


class ProjectorControlledPhaseShift:
    """
    Projector-controlled phase shift circuit.
    Building block for Qubitization.
    2 possible methods: "auxiliary" or "c-phase".
    Projector is state |0>, |00>... on the encoding (auxiliary) qubitS.
    """
    def __init__(self, theta: float=0,
                 projection_state: Sequence[int]=[0],
                 encoding_qubits:  Qubit | Sequence[Qubit]=None,
                 auxiliary_qubits: Qubit | Sequence[Qubit]=None,
                 method = "auxiliary"):
        self.theta = theta
        if not set(projection_state).issubset({0,1}):
            raise ValueError("The projection state can only have entries 1 or 0.")
        self.projection_state = list(projection_state)
        # I check the compaitibility between the sizes of the state and the encoding qubits' list in the 'as_matrix()' method
        if type(encoding_qubits) == Qubit:
            self.encoding_qubits = [encoding_qubits]
        elif encoding_qubits is not None:
            self.encoding_qubits = list(encoding_qubits)
        if method == "auxiliary":
            if type(auxiliary_qubits) == Qubit:
                self.auxiliary_qubits = [auxiliary_qubits]
            elif auxiliary_qubits is not None:
                self.auxiliary_qubits = list(auxiliary_qubits)
        else:
            self.auxiliary_qubits = []
        if method not in ["auxiliary", "c-phase"]:
            raise RuntimeError("The method {method} is not valid. Only use 'auxiliary' or 'c-phase'.")
        self.method = method

    def set_projection_state(self, projection_state: Sequence[int] = [1,0]):
        """
        Set the projection state.
        """
        if set(projection_state) != set([0,1]):
            raise ValueError("The projection state can only have entries 1 or 0.")
        self.projection_state = list(projection_state)

    def set_method(self, method = "auxiliary"):
        """
        Set the method.
        If method == 'auxiliary' the auxiliary qubits' list gets erased.
        """
        if method not in ["auxiliary", "c-phase"]:
            raise RuntimeError("The method {method} is not valid. Only use 'auxiliary' or 'c-phase'.")
        self.method = method
        if method != "auxiliary":
            self.auxiliary_qubits = []

    def set_encoding_qubits(self, *args):
        """
        Set the encoding qubits.
        """
        if len(args) == 1 and isinstance(args[0], Sequence):
            encoding_qubits = list(args[0])
        else:
            encoding_qubits = list(args)
        self.encoding_qubits = encoding_qubits
        # enable chaining
        return self

    def set_auxiliary_qubits(self, *args):
        """
        Set the auxiliary qubits.
        Only works if the method is set to auxiliary.
        """
        if self.method != "auxiliary":
            return self
        if len(args) == 1 and isinstance(args[0], Sequence):
            auxiliary_qubits = list(args[0])
        else:
            auxiliary_qubits = list(args)
        self.auxiliary_qubits = auxiliary_qubits
        # enable chaining
        return self

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        One extra wire with method 'auxiliary'.
        """
        if self.method == "auxiliary":
            return len(self.auxiliary_qubits) + len(self.encoding_qubits)
        else:
            return len(self.encoding_qubits)

    def set_theta(self, theta):
        """
        Set the angle theta.
        """
        self.theta = theta

    def as_matrix(self):
        """
        Generate the matrix representation of the controlled gate.
        The extra wire from 'auxiliary' method is not taken into account.
        """
        size_enc = len(self.projection_state)
        if any(s != 0 for s in self.projection_state):
            raise RuntimeError("The projection state can only have entries equal to 0.")
        binary_index = int(''.join(map(str,self.projection_state)), 2)
        basis_state = np.zeros((2**size_enc))
        basis_state[binary_index] = 1
        matrix = np.outer(basis_state, basis_state)
        mat_ref = expm(1j*self.theta*(2*matrix - np.identity(2**size_enc)))
        return mat_ref

    def as_circuit(self):
        """
        Generates the circuit.
        *** with 'auxiliary' method I have an extra wire.
        *** In order to compare it to as_matrix() you need to compare only the half upper-left block.
        """
        size_enc = len(self.encoding_qubits)
        if len(self.projection_state) != size_enc:
            raise RuntimeError("The size of the projection state must be {2**(len(self.encoding_qubits))} while {len(self.projection_state)} entries were given.")
        if any(s != 0 for s in self.projection_state):
            raise RuntimeError("The projection state can only have entries equal to 0.")
        circuit = Circuit()
        if self.method == "auxiliary":
            multi_cnot = ControlledGate(PauliXGate(self.auxiliary_qubits[0]), size_enc, self.projection_state)
            multi_cnot.set_control(self.encoding_qubits)
            circuit.append_gate(multi_cnot)
            circuit.append_gate(RzGate(2*self.theta, self.auxiliary_qubits[0]))
            circuit.append_gate(multi_cnot)
        else:
            max_den = size_enc-1
            circuit.append_gate(RzGate(-2*self.theta/(2**max_den), self.encoding_qubits[0]))
            for i in range(1, size_enc):
                multi_rot = ControlledGate(RzGate(-2*self.theta/(2**(max_den-i)), self.encoding_qubits[i]), i, self.projection_state[:i])
                multi_rot.set_control(self.encoding_qubits[:i])
                circuit.append_gate(multi_rot)
            # get rid of global phase
            glob_p = PhaseFactorGate((1-2**max_den)*self.theta/2**max_den, self.num_wires)
            glob_p.on(self.encoding_qubits)
            circuit.append_gate(glob_p)
        return circuit