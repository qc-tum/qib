import unittest
import numpy as np
import qib


class TestFullyConnectedLattice(unittest.TestCase):

    def test_lattice_adjacency(self):
        """
        Test construction of adjacency matrices.
        """
        n_list = [5, 7, 10, 15]
        for n in n_list:
            latt = qib.lattice.FullyConnectedLattice(n)
            self.assertEqual(latt.ndim, 1)
            self.assertEqual(latt.nsites, n)
            adj = latt.adjacency_matrix()
            adj_ref = np.ones((n,n))
            for i in range(n):
                adj_ref[i,i] = 0
            self.assertTrue(np.array_equal(adj, adj_ref))

            latt = qib.lattice.FullyConnectedLattice((n,))
            self.assertEqual(latt.ndim, 1)
            self.assertEqual(latt.nsites, n)
            adj = latt.adjacency_matrix()
            self.assertTrue(np.array_equal(adj, adj_ref))

            latt = qib.lattice.FullyConnectedLattice((n,3))
            self.assertEqual(latt.ndim, 2)
            self.assertEqual(latt.nsites, n*3)
            adj = latt.adjacency_matrix()
            adj_ref = np.ones((3*n,3*n))
            for i in range(3*n):
                adj_ref[i,i] = 0
            self.assertTrue(np.array_equal(adj, adj_ref))

    def test_lattice_coords(self):
        """
        Test lattice coordinate indexing.
        """
        # one-dimensional lattice
        for L in range(3, 10):
            latt = qib.lattice.FullyConnectedLattice((L,))
            for i in range(latt.nsites):
                self.assertEqual(i, latt.coord_to_index(latt.index_to_coord(i)))

        # two-dimensional lattice
        for Lx in range(3, 5):
            for Ly in range(3, 5):
                latt = qib.lattice.FullyConnectedLattice((Lx, Ly))
                for i in range(latt.nsites):
                    self.assertEqual(i, latt.coord_to_index(latt.index_to_coord(i)))

        # three-dimensional lattice
        for Lx in range(3, 5):
            for Ly in range(3, 5):
                for Lz in range(3, 5):
                    latt = qib.lattice.FullyConnectedLattice((Lx, Ly, Lz))
                    for i in range(latt.nsites):
                        self.assertEqual(i, latt.coord_to_index(latt.index_to_coord(i)))


if __name__ == "__main__":
    unittest.main()
