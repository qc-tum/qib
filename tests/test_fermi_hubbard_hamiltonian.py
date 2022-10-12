import numpy as np
from scipy import sparse
import unittest
import qib


class TestFermiHubbardHamiltonian(unittest.TestCase):

    def test_construction(self):
        """
        Test construction of the Fermi-Hubbard Hamiltonian.
        """
        # uniform parameters, integer lattice
        L = 5
        latt = qib.lattice.IntegerLattice((L,), pbc=False)
        field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
        adj = latt.adjacency_matrix()
        t = np.random.uniform(-5,5)
        u = np.random.uniform(-5,5)
        # construct Hamiltonian
        H = qib.operator.FermiHubbardHamiltonian(field, t, u)
        self.assertEqual(H.fields(), [field])
        self.assertTrue(H.is_hermitian())
        Hmat = H.as_matrix()
        # must be symmetric
        self.assertAlmostEqual(sparse.linalg.norm(Hmat - Hmat.conj().T), 0)
        # only nearest neighbours and symmetric for hopping term
        op = H.as_field_operator().terms
        self.assertTrue(len(op) == 2)
        for i in range(latt.nsites):
            for j in range(i+1, latt.nsites):
                if adj[i,j] == 0:
                    self.assertTrue(op[0].coeffs[i,j] == 0)
                self.assertTrue(op[0].coeffs[i,j] == op[0].coeffs[j,i])
            self.assertTrue(op[0].coeffs[i,i] == 0)
        
        # uniform parameters, hexagonal lattice
        Lx, Ly = (3,5)
        latt = qib.lattice.HexagonalLattice((Lx,Ly), pbc=False)
        field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
        adj = latt.adjacency_matrix()
        t = np.random.uniform(-5,5)
        u = np.random.uniform(-5,5)
        # construct Hamiltonian
        H = qib.operator.FermiHubbardHamiltonian(field, t, u)
        self.assertEqual(H.fields(), [field])
        self.assertTrue(H.is_hermitian())

        # only nearest neighbours and symmetric for hopping term
        op = H.as_field_operator().terms
        self.assertTrue(len(op) == 2)
        for i in range(latt.nsites):
            for j in range(i+1, latt.nsites):
                if adj[i,j] == 0:
                    self.assertTrue(op[0].coeffs[i,j] == 0)
                self.assertTrue(op[0].coeffs[i,j] == op[0].coeffs[j,i])
            self.assertTrue(op[0].coeffs[i,i] == 0)

if __name__ == "__main__":
    unittest.main()
