import numpy as np
from typing import Sequence, Union
from qib.field import Qubit
from qib.operator import ProjectorControlledPhaseShift, BlockEncodingGate, BlockEncodingMethod
from qib.circuit import Circuit



class EigenvalueTransformation:
    """
    Eigenvalue transformation for a given unitary (encoding).
    It requires the unitary gate that gets processed, the projector-controlled phase shift and the list of angles for the processing.
    TODO: generalize with different projectors
    TODO: generalize with longer arrays of qubits
    """
    def __init__(self, block_encoding, processing_gate, theta_seq: Sequence[float]=None):
        self.block_encoding = block_encoding
        self.processing_gate = processing_gate

        if theta_seq is not None:
            self.theta_seq = list(theta_seq)
        else:
            self.theta_seq = theta_seq

    def set_theta_seq(self, theta_seq: Sequence[float]):
        """
        Set the angles theta for the eigenvalue transformation
        """
        if theta_seq is not None:
            self.theta_seq = list(theta_seq)
        else:
            self.theta_seq = theta_seq

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        TODO: generalize when using a larger state as projector
        """
        return self.block_encoding.num_wires + 1

    def set_auxiliary_qubits(self, q_anc: Union[Qubit, Sequence[Qubit]]):
        """
        Set the auxiliary qubits.
        """
        self.processing_gate.set_auxiliary_qubits(q_anc)

    def set_encoding_qubits(self, q_enc: Union[Qubit, Sequence[Qubit]]):
        """
        Set the encoding extra qubits
        """
        self.processing_gate.set_encoding_qubits(q_enc)
        self.block_encoding.set_auxiliary_qubits(q_enc)

    def as_matrix(self):
        """
        Generate the matrix representation of the eigenvalue transformation.
        Format: |ancillary_Pi> @ |enc_extra> @ |encoded_state>
        """
        if self.block_encoding.auxiliary_qubits != self.processing_gate.encoding_qubits:
            raise RuntimeError("The block encoding's auxiliary qubits and processing gate's encoding extra qubits must be the same")
        if not self.theta_seq:
            raise ValueError("the angles 'theta' have not been initialized")
        matrix = np.identity(2**self.num_wires)
        num_particles = self.block_encoding.num_wires - self.block_encoding.num_aux_qubits
        id_for_projector = np.identity(2**num_particles)
        id_for_unitary = np.identity(2**len(self.processing_gate.auxiliary_qubits))
        U_inv_matrix = self.block_encoding.inverse().as_matrix()
        U_matrix = self.block_encoding.as_matrix()
        if(len(self.theta_seq)%2==0):
            dim = len(self.theta_seq)//2
            start = 0
        else:
            dim = (len(self.theta_seq)-1)//2
            self.processing_gate.set_theta(self.theta_seq[0])
            matrix = matrix @ np.kron(self.processing_gate.as_matrix(), id_for_projector) \
                            @ np.kron(id_for_unitary, U_matrix)
            start = 1
        for i in range(start, dim):
            self.processing_gate.set_theta(self.theta_seq[2*i-start])
            matrix = matrix @ np.kron(self.processing_gate.as_matrix(), id_for_projector) \
                            @ np.kron(id_for_unitary, U_inv_matrix) 
            self.processing_gate.set_theta(self.theta_seq[2*i+1-start])
            matrix = matrix @ np.kron(self.processing_gate.as_matrix(), id_for_projector) \
                            @ np.kron(id_for_unitary, U_matrix)
        return matrix

    def as_circuit(self):
        """
        Generates the qubitization circuit
        """
        if self.block_encoding.auxiliary_qubits != self.processing_gate.encoding_qubits:
            raise RuntimeError("The block encoding's auxiliary qubits and processing gate's encoding qubits must be the same")
        if not self.theta_seq:
            raise ValueError("the angles 'theta' have not been initialized")
        circuit = Circuit()

        if(len(self.theta_seq)%2==0):
            dim = len(self.theta_seq)//2
            start = 0
        else:
            dim = (len(self.theta_seq)-1)//2
            self.processing_gate.set_theta(self.theta_seq[0])
            circuit.prepend_gate(self.processing_gate)
            circuit.prepend_gate(self.block_encoding)
            start = 1
        for i in range(start, dim):
            self.processing_gate.set_theta(self.theta_seq[2*i-start])
            circuit.prepend_gate(self.processing_gate)
            circuit.prepend_gate(self.block_encoding.inverse())
            self.processing_gate.set_theta(self.theta_seq[2*i+1-start])
            circuit.prepend_gate(self.processing_gate)
            circuit.prepend_gate(self.block_encoding)
        return circuit

