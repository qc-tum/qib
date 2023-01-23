import numpy as np
from scipy.linalg import expm
import unittest
import qib



class TestQubitization(unittest.TestCase):
    
    def test_projector_controlled_phase_shift(self):
        """
        Test the projector controlled phase shift
        """
        # encoding qubits' field
        field1 = qib.field.Field(qib.field.ParticleType.QUBIT,
                                 qib.lattice.IntegerLattice((4,), pbc=False))
        # auxiliary qubit
        field2 = qib.field.Field(qib.field.ParticleType.QUBIT,
                                 qib.lattice.IntegerLattice((1,), pbc=False))
        q_aux = [qib.field.Qubit(field2, 0)]
        for i, method in enumerate(["c-phase", "auxiliary"]):
            for state in [[0], [0,0], [0,0,0], [0,0,0,0]]:
                # len(state) == number of encoding qubits
                q_enc = [qib.field.Qubit(field1, i) for i in range(len(state))]
                processing = qib.algorithms.qubitization.ProjectorControlledPhaseShift(0., state, q_enc, q_aux, method)
                # must return identity
                processing.set_theta(0.)
                self.assertTrue(np.allclose(processing.as_matrix(), np.identity(2**(len(state)))))
                # must return -identity
                processing.set_theta(np.pi)
                self.assertTrue(np.allclose(processing.as_matrix(), -np.identity(2**(len(state)))))
                # random angle, comparison with definition
                theta = np.random.uniform(0, 2*np.pi)
                processing.set_theta(theta)
                binary_index = int(''.join(map(str,state)), 2)
                basis_state = np.zeros((2**len(state)))
                basis_state[binary_index] = 1
                matrix = np.outer(basis_state, basis_state)
                mat_ref = expm(1j*theta*(2*matrix - np.identity(2**len(state))))
                mat_proc = processing.as_matrix()
                # only upper left block of the auxiliary case can be compared
                self.assertTrue(np.allclose(processing.as_circuit().as_matrix([field2, field1]).toarray()[:2**(5-i), :2**(5-i)],
                                            np.kron(np.kron(np.identity(2**(1-i)), mat_proc), np.identity(2**(4-len(state))))))
                self.assertTrue(np.allclose(mat_proc, mat_ref))
                # check particles, wires and fields
                self.assertTrue(processing.num_wires==len(state)+i)   
                #print("method: ", method, "/ state: ", state, "\t...OK!")

    def test_eigenvalue_transformation(self):
        """
        Test the eigenvalue transformation
        """
        # construct a simple Hamiltonian
        latt = qib.lattice.IntegerLattice((5,), pbc=True)
        field1 = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
        H = qib.operator.HeisenbergHamiltonian(field1, np.random.standard_normal(size=3),
                                                       np.random.standard_normal(size=3))
        # rescale parameters (effectively rescales overall Hamiltonian)
        scale = 1.25 * np.linalg.norm(H.as_matrix().toarray(), ord=2)
        H.J /= scale
        H.h /= scale
        # auxiliary qubits
        # Note: if I choose a bigger lattice the circuit will hold all of it (comparison fails)
        field2 = qib.field.Field(qib.field.ParticleType.QUBIT,
                                 qib.lattice.IntegerLattice((2,), pbc=False))
        q_enc = qib.field.Qubit(field2, 1)
        q_anc = qib.field.Qubit(field2, 0)
        for method_enc in qib.operator.BlockEncodingMethod:
            block = qib.BlockEncodingGate(H, method_enc)
            block.set_auxiliary_qubits(q_enc)
            for j, method_proc in enumerate(["c-phase", "auxiliary"]):
                processing = qib.algorithms.qubitization.ProjectorControlledPhaseShift(0., [0], q_enc, q_anc, method_proc)
                self.assertTrue(block.auxiliary_qubits == processing.encoding_qubits)
                theta = [0.]
                eigen_transform = qib.algorithms.qubitization.EigenvalueTransformation(block,
                                                                                       processing,
                                                                                       theta_seq=theta)
                # obtain block unitary for theta = 0
                self.assertTrue(np.allclose(np.kron(np.identity(2), block.as_matrix()),
                                            eigen_transform.as_circuit().as_matrix([field2, field1]).toarray()))
                # obtain identity for theta == [0, 0]
                theta = [0, 0]
                eigen_transform.set_theta_seq(theta)
                self.assertTrue(np.allclose(np.identity(2**(latt.nsites + field2.lattice.nsites)),
                                            eigen_transform.as_circuit().as_matrix([field2, field1]).toarray()))
                # random theta
                theta = [np.random.uniform(0, 2*np.pi) for i in range(5)]
                for i in range(1, 5):
                    eigen_transform.set_theta_seq(theta[:i])
                    mat_class = eigen_transform.as_matrix()
                    circ_class = eigen_transform.as_circuit().as_matrix([field2, field1]).toarray()[:2**6, :2**6]
                    self.assertTrue(np.allclose(mat_class, circ_class))
    
                # TODO: add more tests
                #print("method encoding: ", method_enc, "method processing: ", method_proc, "/ state: ", [0], "\t...OK!")

if __name__ == "__main__":
     unittest.main()
