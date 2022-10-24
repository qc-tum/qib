import numpy as np
import unittest
import qib


class TestFermiHubbardHamiltonian(unittest.TestCase):

    def test_construction_no_spin(self):
        """
        Test construction of the Fermi-Hubbard Hamiltonian.
        No spin.
        """
        # uniform parameters, integer lattice
        Lx, Ly = (3,5)
        latt = qib.lattice.IntegerLattice((Lx,Ly), pbc=False)
        field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
        adj = latt.adjacency_matrix()
        t = np.random.uniform(-5,5)
        u = np.random.uniform(-5,5)
        # construct Hamiltonian
        H = qib.operator.FermiHubbardHamiltonian(field, t, u, spin=False)
        self.assertEqual(H.fields(), [field])
        self.assertTrue(H.is_hermitian())
        #Hmat = H.as_matrix()
        # only nearest neighbours and symmetric for hopping term
        op = H.as_field_operator().terms
        self.assertTrue(len(op) == 2)
        for i in range(latt.nsites):
            for j in range(i+1, latt.nsites):
                if adj[i,j] == 0:
                    self.assertTrue(op[0].coeffs[i,j] == 0)
                else:
                    self.assertTrue(op[0].coeffs[i,j] == t)
                self.assertTrue(op[0].coeffs[i,j] == op[0].coeffs[j,i])
            self.assertTrue(op[0].coeffs[i,i] == 0)
            self.assertTrue(op[1].coeffs[i,i] == u)

        # uniform parameters, hexagonal lattice
        Lx, Ly = (3,5)
        latt = qib.lattice.HexagonalLattice((Lx,Ly), pbc=False)
        field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
        adj = latt.adjacency_matrix()
        t = np.random.uniform(-5,5)
        u = np.random.uniform(-5,5)
        # construct Hamiltonian
        H = qib.operator.FermiHubbardHamiltonian(field, t, u, spin=False)
        self.assertEqual(H.fields(), [field])
        self.assertTrue(H.is_hermitian())
        # only nearest neighbours and symmetric for hopping term
        op = H.as_field_operator().terms
        self.assertTrue(len(op) == 2)
        for i in range(latt.nsites):
            for j in range(i+1, latt.nsites):
                if adj[i,j] == 0:
                    self.assertTrue(op[0].coeffs[i,j] == 0)
                else:
                    self.assertTrue(op[0].coeffs[i,j] == t)
                self.assertTrue(op[0].coeffs[i,j] == op[0].coeffs[j,i])
            self.assertTrue(op[0].coeffs[i,i] == 0)
            self.assertTrue(op[1].coeffs[i,i] == u)

    def test_construction_with_spin(self):
        """
        Test construction of the Fermi-Hubbard Hamiltonian.
        With spin.
        """
        # uniform parameters, integer lattice
        Lx, Ly = (3,5)
        latt = qib.lattice.IntegerLattice((Lx,Ly), pbc=False)
        field = qib.field.Field(qib.field.ParticleType.FERMION, qib.lattice.LayeredLattice(latt, 2))
        adj = latt.adjacency_matrix()
        t = np.random.uniform(-5,5)
        u = np.random.uniform(-5,5)
        t_matrix = np.kron(np.identity(2), adj)
        # construct Hamiltonian
        H = qib.operator.FermiHubbardHamiltonian(field, t, u, spin=True)
        self.assertEqual(H.fields(), [field])
        self.assertTrue(H.is_hermitian())
        #Hmat = H.as_matrix()
        # only nearest neighbours and symmetric for hopping term
        op = H.as_field_operator().terms
        self.assertTrue(len(op) == 2)
        for i in range(latt.nsites):
            for j in range(latt.nsites):
                for k in range(latt.nsites):
                    for l in range(latt.nsites):
                        if i==j and j==k and k==l:
                            self.assertTrue(op[1].coeffs[i,j,k,l] == u)
                        else:
                            self.assertTrue(op[1].coeffs[i,j,k,l] == 0)
                if t_matrix[i,j] == 0:
                    self.assertTrue(op[0].coeffs[i,j] == 0)
                else:
                    self.assertTrue(op[0].coeffs[i,j] == t)
                self.assertTrue(op[0].coeffs[i,j] == op[0].coeffs[j,i])
            self.assertTrue(op[0].coeffs[i,i] == 0)

        # uniform parameters, hexagonal lattice
        Lx, Ly = (3,5)
        latt = qib.lattice.HexagonalLattice((3,5), pbc=False)
        field = qib.field.Field(qib.field.ParticleType.FERMION, qib.lattice.LayeredLattice(latt, 2))
        adj = latt.adjacency_matrix()
        t = np.random.uniform(-5,5)
        u = np.random.uniform(-5,5)
        t_matrix = np.kron(np.identity(2), adj)
        # construct Hamiltonian
        H = qib.operator.FermiHubbardHamiltonian(field, t, u, spin=True)
        self.assertEqual(H.fields(), [field])
        self.assertTrue(H.is_hermitian())
        # only nearest neighbours and symmetric for hopping term
        op = H.as_field_operator().terms
        self.assertTrue(len(op) == 2)
        for i in range(latt.nsites):
            for j in range(latt.nsites):
                for k in range(latt.nsites):
                    for l in range(latt.nsites):
                        if i==j and j==k and k==l:
                            self.assertTrue(op[1].coeffs[i,j,k,l] == u)
                        else:
                            self.assertTrue(op[1].coeffs[i,j,k,l] == 0)
                if t_matrix[i,j] == 0:
                    self.assertTrue(op[0].coeffs[i,j] == 0)
                else:
                    self.assertTrue(op[0].coeffs[i,j] == t)
                self.assertTrue(op[0].coeffs[i,j] == op[0].coeffs[j,i])
            self.assertTrue(op[0].coeffs[i,i] == 0)


if __name__ == "__main__":
    unittest.main()
