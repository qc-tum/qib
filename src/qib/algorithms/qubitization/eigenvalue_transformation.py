import numpy as np
from scipy.sparse import csr_matrix
from typing import Sequence
from qib.field import Field, Particle, Qubit
from qib.operator import ProjectorControlledPhaseShift, BlockEncodingGate, BlockEncodingMethod
from qib.circuit import Circuit
from qib.operator import AbstractOperator


class EigenvalueTransformation:
    """
    Eigenvalue transformation for a given unitary (encoding).
    It requires the unitary gate that gets processed, the projector-controlled phase shift and the list of angles for the processing.
    TODO: generalize with different projectors
    TODO: generalize with longer arrays of qubits
    """
    def __init__(self, h, method: BlockEncodingMethod, q_enc: Qubit, q_anc: Qubit, projector: Sequence[float]=[1,0], theta_seq: Sequence[float]=None):
        if len(projector)!=2 or projector[0] != 1 or projector[1] != 0:
            raise ValueError("projector can only be the |0> state --> vector (1, 0)")
            
        self.block_encoding = BlockEncodingGate(h, method)
        self.processing_gate = ProjectorControlledPhaseShift()
        # The block encoding auxiliary qubit and the processing gate "encoding qubit"should be the same
        self.block_encoding.set_auxiliary_qubits([q_enc])
        self.processing_gate.set_encoding_qubits([q_enc])
        self.processing_gate.set_auxiliary_qubits([q_anc])

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

    def as_matrix(self):
        """
        Generate the matrix representation of the eigenvalue transformation
        """
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