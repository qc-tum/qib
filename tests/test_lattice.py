import numpy as np
import unittest
import qib


class TestLattice(unittest.TestCase):

    def test_integer_lattice_adjacency(self):
        """
        Test construction of adjacency matrices for an integer lattice.
        """

        # one-dimensional lattice, periodic boundary conditions
        for L in range(3, 10):
            latt = qib.lattice.IntegerLattice((L,), pbc=True)
            self.assertEqual(latt.ndim, 1)
            self.assertEqual(latt.nsites, L)
            adj = latt.adjacency_matrix()
            self.assertTrue(np.array_equal(adj, adj.T))
            adj_ref = np.roll(np.identity(L), 1, axis=0) + np.roll(np.identity(L), -1, axis=0)
            self.assertTrue(np.array_equal(adj, adj_ref))

        # one-dimensional lattice, open boundary conditions
        for L in range(3, 10):
            latt = qib.lattice.IntegerLattice((L,), pbc=False)
            self.assertEqual(latt.ndim, 1)
            self.assertEqual(latt.nsites, L)
            adj = latt.adjacency_matrix()
            self.assertTrue(np.array_equal(adj, adj.T))
            adj_ref = np.diag(np.ones(L - 1, dtype=int), k=1) + np.diag(np.ones(L - 1, dtype=int), k=-1)
            self.assertTrue(np.array_equal(adj, adj_ref))

        for Lx in range(3, 5):
            for Ly in range(3, 5):
                # two-dimensional lattice, periodic boundary conditions
                latt = qib.lattice.IntegerLattice((Lx, Ly), pbc=True)
                self.assertEqual(latt.ndim, 2)
                self.assertEqual(latt.nsites, Lx*Ly)
                adj = latt.adjacency_matrix()
                self.assertTrue(np.array_equal(adj, adj.T))
                self.assertTrue(np.array_equal(np.sum(adj, axis=0), latt.nsites*[4]))
                self.assertTrue(np.array_equal(np.sum(adj, axis=1), latt.nsites*[4]))
                enum_latt = [(x, y) for x in range(Lx) for y in range(Ly)]
                adj_ref = np.zeros((len(enum_latt), len(enum_latt)), dtype=int)
                for i, a in enumerate(enum_latt):
                    for j, b in enumerate(enum_latt):
                        if abs((a[0] - b[0] + 1) % Lx - 1) + abs((a[1] - b[1] + 1) % Ly - 1) == 1:
                            adj_ref[i, j] = 1
                self.assertTrue(np.array_equal(adj, adj_ref))

                # two-dimensional lattice, open boundary conditions
                latt = qib.lattice.IntegerLattice((Lx, Ly), pbc=False)
                self.assertEqual(latt.ndim, 2)
                self.assertEqual(latt.nsites, Lx*Ly)
                adj = latt.adjacency_matrix()
                self.assertTrue(np.array_equal(adj, adj.T))
                enum_latt = [(x, y) for x in range(Lx) for y in range(Ly)]
                adj_ref = np.zeros((len(enum_latt), len(enum_latt)), dtype=int)
                for i, a in enumerate(enum_latt):
                    for j, b in enumerate(enum_latt):
                        if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                            adj_ref[i, j] = 1
                self.assertTrue(np.array_equal(adj, adj_ref))

                # two-dimensional lattice, open boundary conditions in
                # x-direction and periodic boundary conditions in y-direction
                latt = qib.lattice.IntegerLattice((Lx, Ly), pbc=(False, True))
                self.assertEqual(latt.ndim, 2)
                self.assertEqual(latt.nsites, Lx*Ly)
                adj = latt.adjacency_matrix()
                self.assertTrue(np.array_equal(adj, adj.T))
                enum_latt = [(x, y) for x in range(Lx) for y in range(Ly)]
                adj_ref = np.zeros((len(enum_latt), len(enum_latt)), dtype=int)
                for i, a in enumerate(enum_latt):
                    for j, b in enumerate(enum_latt):
                        if abs(a[0] - b[0]) + abs((a[1] - b[1] + 1) % Ly - 1) == 1:
                            adj_ref[i, j] = 1
                self.assertTrue(np.array_equal(adj, adj_ref))

    def test_integer_lattice_coords(self):
        """
        Test coordinate indexing.
        """
        # one-dimensional lattice
        for L in range(3, 10):
            latt = qib.lattice.IntegerLattice((L,))
            for i in range(latt.nsites):
                self.assertEqual(i, latt.coord_to_index(latt.index_to_coord(i)))

        # two-dimensional lattice
        for Lx in range(3, 5):
            for Ly in range(3, 5):
                latt = qib.lattice.IntegerLattice((Lx, Ly))
                for i in range(latt.nsites):
                    self.assertEqual(i, latt.coord_to_index(latt.index_to_coord(i)))

        # three-dimensional lattice
        for Lx in range(3, 5):
            for Ly in range(3, 5):
                for Lz in range(3, 5):
                    latt = qib.lattice.IntegerLattice((Lx, Ly, Lz))
                    for i in range(latt.nsites):
                        self.assertEqual(i, latt.coord_to_index(latt.index_to_coord(i)))


if __name__ == '__main__':
    unittest.main()
