import numpy as np
from typing import Sequence, Union
from qib.field import Qubit
from qib.operator import Gate
from qib.algorithms.qubitization.projector_controlled_phase_shift import ProjectorControlledPhaseShift
from qib.circuit import Circuit


class EigenvalueTransformation:
    """
    Eigenvalue transformation for a given unitary (encoding).
    It requires the unitary gate that gets processed, the projector-controlled phase shift and the list of angles for the processing.
    """
    def __init__(self, block_encoding: Gate, processing: ProjectorControlledPhaseShift, theta_seq: Sequence[float]=None):
        self.block_encoding = block_encoding
        self.processing = processing

        if theta_seq is not None:
            self.theta_seq = list(theta_seq)
        else:
            self.theta_seq = theta_seq

    def set_theta_seq(self, theta_seq: Sequence[float]):
        """
        Set the angles theta for the eigenvalue transformation.
        """
        if theta_seq is not None:
            self.theta_seq = list(theta_seq)
        else:
            self.theta_seq = theta_seq

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return self.block_encoding.num_wires + len(self.processing.auxiliary_qubits)

    def set_auxiliary_qubits(self, q_anc: Union[Qubit, Sequence[Qubit]]):
        """
        Set the auxiliary qubits.
        """
        self.processing.set_auxiliary_qubits(q_anc)
    
    def set_projection_state(self, projection_state: Sequence[int] = [1,0]):
        """
        Set the projection state.
        """
        self.processing.set_projection_state(projection_state)

    def set_method(self, method = "auxiliary"):
        """
        Set the method.
        """
        self.processing.set_method(method)

    def set_encoding_qubits(self, q_enc: Union[Qubit, Sequence[Qubit]]):
        """
        Set the encoding extra qubits.
        """
        self.processing.set_encoding_qubits(q_enc)
        self.block_encoding.set_auxiliary_qubits(q_enc)

    def as_matrix(self):
        """
        Generate the matrix representation of the eigenvalue transformation.
        Format: |enc_extra> x |encoded_state>
        Auxiliary wire from 'auxiliary' method is not taken into account.
        """
        if not self.theta_seq:
            raise ValueError("the angles 'theta' have not been initialized.")
        matrix = np.identity(2**self.block_encoding.num_wires)
        num_particles = self.block_encoding.num_wires - self.block_encoding.num_aux_qubits
        id_for_projector = np.identity(2**num_particles)
        U_inv_matrix = self.block_encoding.inverse().as_matrix()
        U_matrix = self.block_encoding.as_matrix()
        if len(self.theta_seq) % 2 == 0:
            dim = len(self.theta_seq) // 2
            start = 0
        else:
            dim = (len(self.theta_seq)-1) // 2
            self.processing.set_theta(self.theta_seq[0])
            matrix = matrix @ np.kron(self.processing.as_matrix(), id_for_projector) \
                            @ U_matrix
            start = 1
        for i in range(start, dim):
            self.processing.set_theta(self.theta_seq[2*i-start])
            matrix = matrix @ np.kron(self.processing.as_matrix(), id_for_projector) \
                            @ U_inv_matrix
            self.processing.set_theta(self.theta_seq[2*i+1-start])
            matrix = matrix @ np.kron(self.processing.as_matrix(), id_for_projector) \
                            @ U_matrix
        return matrix

    def as_circuit(self):
        """
        Generates the qubitization circuit.
        *** with 'auxiliary' method I have an extra wire.
        *** In order to compare it to as_matrix() you need to compare only the half upper-left block.
        """
        if self.block_encoding.auxiliary_qubits != self.processing.encoding_qubits:
            raise RuntimeError("The block encoding's auxiliary qubits and processing gate's encoding qubits must be the same.")
        if not self.theta_seq:
            raise ValueError("the angles 'theta' have not been initialized.")
        circuit = Circuit()

        if len(self.theta_seq) % 2 == 0:
            dim = len(self.theta_seq) // 2
            start = 0
        else:
            dim = (len(self.theta_seq)-1)//2
            self.processing.set_theta(self.theta_seq[0])
            circuit.prepend_circuit(self.processing.as_circuit())
            circuit.prepend_gate(self.block_encoding)
            start = 1
        for i in range(start, dim):
            self.processing.set_theta(self.theta_seq[2*i-start])
            circuit.prepend_circuit(self.processing.as_circuit())
            circuit.prepend_gate(self.block_encoding.inverse())
            self.processing.set_theta(self.theta_seq[2*i+1-start])
            circuit.prepend_circuit(self.processing.as_circuit())
            circuit.prepend_gate(self.block_encoding)
        return circuit
