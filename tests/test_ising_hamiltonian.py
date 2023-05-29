import unittest
import numpy as np
from scipy import sparse
import qib


class TestIsingHamiltonian(unittest.TestCase):

    def test_construction(self):
        """
        Test construction of the Ising model Hamiltonian.
        """
        L = 5
        # Hamiltonian parameters
        J =  5.0/11
        h = -2.0/7
        g = 13.0/8
        # construct Hamiltonian
        latt = qib.lattice.IntegerLattice((L,), pbc=False)
        field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
        H = qib.IsingHamiltonian(field, J, h, g)
        self.assertEqual(H.fields(), [field])
        self.assertTrue(H.is_hermitian())
        Hmat = H.as_matrix()
        # must be symmetric
        self.assertAlmostEqual(sparse.linalg.norm(Hmat - Hmat.conj().T), 0)
        # reference Hamiltonian
        X = np.array([[ 0.,  1.], [ 1.,  0.]])
        Z = np.array([[ 1.,  0.], [ 0., -1.]])
        Href = 0
        for i in range(L - 1):
            Href += J * np.kron(np.identity(2**i), np.kron(np.kron(Z, Z), np.identity(2**(L-i-2))))
        for i in range(L):
            Href += np.kron(np.identity(2**i), np.kron(h * Z + g * X, np.identity(2**(L-i-1))))
        # compare
        self.assertTrue(np.allclose(H.as_matrix().toarray(), Href))


if __name__ == "__main__":
    unittest.main()
