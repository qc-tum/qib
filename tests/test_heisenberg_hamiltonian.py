import unittest
import numpy as np
from scipy import sparse
import qib


class TestHeisenbergHamiltonian(unittest.TestCase):

    def test_construction(self):
        """
        Test construction of the Heisenberg model Hamiltonian.
        """
        rng = np.random.default_rng()
        L = 5
        # Hamiltonian parameters
        J = [rng.uniform(-5, 5) for i in range(3)]
        h = [rng.uniform(-5, 5) for i in range(3)]
        # construct Hamiltonian
        latt = qib.lattice.IntegerLattice((L,), pbc=False)
        field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
        H = qib.operator.HeisenbergHamiltonian(field, J, h)
        self.assertEqual(H.fields(), [field])
        self.assertTrue(H.is_hermitian())
        Hmat = H.as_matrix()
        # must be symmetric
        self.assertAlmostEqual(sparse.linalg.norm(Hmat - Hmat.conj().T), 0)
        # reference Hamiltonian
        X = np.array([[ 0.,  1.], [ 1.,  0.]])
        Y = np.array([[ 0., -1j], [ 1j,  0.]])
        Z = np.array([[ 1.,  0.], [ 0., -1.]])
        Href = 0j
        for k, gate in enumerate([X, Y, Z]):
            for i in range(L-1):
                Href += J[k] * np.kron(np.identity(2**i), np.kron(np.kron(gate, gate), np.identity(2**(L-i-2))))
            for i in range(L):
                Href += np.kron(np.identity(2**i), np.kron(h[k]*gate, np.identity(2**(L-i-1))))
        # compare
        self.assertTrue(np.allclose(H.as_matrix().toarray(), Href))


if __name__ == "__main__":
    unittest.main()
