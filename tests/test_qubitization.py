import numpy as np
import unittest
import qib



class TestQubitization(unittest.TestCase):

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
        for method in qib.operator.BlockEncodingMethod:
            block = qib.BlockEncodingGate(H, method)
            block.set_auxiliary_qubits(q_enc)
            processing = qib.ProjectorControlledPhaseShift(0., q_enc, q_anc)
            self.assertTrue(block.auxiliary_qubits == processing.encoding_qubits)
            theta = [0.]
            eigen_transform = qib.algorithms.qubitization.EigenvalueTransformation(block,
                                                                                   processing,
                                                                                   theta_seq=theta)
            eigen_transform_gate = qib.EigenvalueTransformationGate(block, processing, theta)
            # obtain block unitary for theta = 0
            self.assertTrue(np.allclose(np.kron(np.identity(2**1), block.as_matrix()),
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
                circ_class = eigen_transform.as_circuit().as_matrix([field2, field1]).toarray()
                self.assertTrue(np.allclose(mat_class, circ_class))

            # TODO: add more tests


if __name__ == "__main__":
     unittest.main()
