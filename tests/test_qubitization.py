import numpy as np
import unittest
import qib


class TestQubitization(unittest.TestCase):

    def test_eigenvalue_transformation(self):
        """
        Test the eigenvalue transformation
        """
        # construct a simple Hamiltonian
        L = 2
        latt = qib.lattice.IntegerLattice((L,), pbc=True)
        field1 = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
        #np.random.seed(0)
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
        q_enc = qib.field.Qubit(field2, 0)
        q_anc = qib.field.Qubit(field2, 1)
        for method in qib.operator.BlockEncodingMethod:
            block = qib.BlockEncodingGate(H, method)
            block.set_auxiliary_qubits(q_enc)
            processing = qib.ProjectorControlledPhaseShift(q_enc, q_anc)
            self.assertTrue(block.auxiliary_qubits == processing.encoding_qubits)
            theta = [0.]
            eigen_transform = qib.algorithms.qubitization.EigenvalueTransformation(h=H, 
                                                                                   method=method, 
                                                                                   q_enc=q_enc,
                                                                                   q_anc=q_anc, 
                                                                                   projector=[1,0], 
                                                                                   theta_seq=theta)
            eigen_transform_gate = qib.EigenvalueTransformationGate(block, processing, theta)
            # if theta == 0 I only have block unitary
            self.assertTrue(np.allclose(np.kron(np.identity(2**1), block.as_matrix()), eigen_transform.as_circuit().as_matrix([field1,field2]).toarray()))
            # if theta == [0,0] I have identity
            theta = [0,0]
            eigen_transform.set_theta_seq(theta)
            self.assertTrue(np.allclose(np.identity(2**(latt.nsites+field2.lattice.nsites)), eigen_transform.as_circuit().as_matrix([field1,field2]).toarray()))
            # random theta
            theta = [np.random.uniform(0, 2*np.pi) for i in range(10)]
            for i in range(1,10):
                eigen_transform.set_theta_seq(theta[:i])
                eigen_transform_gate.set_theta_seq(theta[:i])
                mat_class = eigen_transform.as_matrix()
                mat_gate = eigen_transform_gate.as_matrix()
                circ_class = eigen_transform.as_circuit().as_matrix([field1,field2]).toarray()
                circ_gate = eigen_transform_gate._circuit_matrix([field1, field2]).toarray()
                self.assertTrue(np.allclose(mat_class, mat_gate))
                self.assertTrue(np.allclose(mat_class, circ_gate))
                self.assertTrue(np.allclose(mat_class, circ_class))
            # TODO: add more tests


if __name__ == "__main__":
    unittest.main()
