import math
import unittest
import numpy as np
import qib


class TestCustomizedLattice(unittest.TestCase):

    def test_lattice_adjacency(self):
        """
        Test construction of adjacency matrices.
        """
        rng = np.random.default_rng()

        n_list = [5, 7, 10, 15]
        for n in n_list:
            shape = (n,)
            nsites = math.prod(shape)
            adj_ref = rng.integers(-10, 10, (nsites, nsites))
            adj_ref += adj_ref.T
            for i in range(nsites):
                adj_ref[i,i] = 0
            latt = qib.lattice.CustomizedLattice(shape, adj_ref)
            self.assertEqual(latt.ndim, 1)
            self.assertEqual(latt.nsites, nsites)
            adj = latt.adjacency_matrix()
            self.assertTrue(np.array_equal(adj, np.array(adj_ref, dtype=bool)))

            shape = (n,n)
            nsites = math.prod(shape)
            adj_ref = rng.integers(-10, 10, (nsites, nsites))
            adj_ref += adj_ref.T
            for i in range(nsites):
                adj_ref[i,i] = 0
            latt = qib.lattice.CustomizedLattice(shape, adj_ref)
            self.assertEqual(latt.ndim, 2)
            self.assertEqual(latt.nsites, nsites)
            adj = latt.adjacency_matrix()
            self.assertTrue(np.array_equal(adj, np.array(adj_ref, dtype=bool)))

            shape = (n, 2, 3)
            nsites = math.prod(shape)
            adj_ref = rng.integers(-10, 10, (nsites, nsites))
            adj_ref += adj_ref.T
            for i in range(nsites):
                adj_ref[i,i] = 0
            latt = qib.lattice.CustomizedLattice(shape, adj_ref)
            self.assertEqual(latt.ndim, 3)
            self.assertEqual(latt.nsites, nsites)
            adj = latt.adjacency_matrix()
            self.assertTrue(np.array_equal(adj, np.array(adj_ref, dtype=bool)))

    def test_lattice_coords(self):
        """
        Test lattice coordinate indexing.
        """
        rng = np.random.default_rng()

        # one-dimensional lattice
        for L in range(3, 10):
            shape = (L,)
            nsites = math.prod(shape)
            adj_ref = rng.integers(-10, 10, (nsites, nsites))
            adj_ref += adj_ref.T
            for i in range(nsites):
                adj_ref[i,i] = 0
            latt = qib.lattice.CustomizedLattice(shape, adj_ref)
            for i in range(latt.nsites):
                self.assertEqual(i, latt.coord_to_index(latt.index_to_coord(i)))

        # two-dimensional lattice
        for Lx in range(3, 5):
            for Ly in range(3, 5):
                shape = (Lx, Ly)
                nsites = math.prod(shape)
                adj_ref = rng.integers(-10, 10, (nsites, nsites))
                adj_ref += adj_ref.T
                for i in range(nsites):
                    adj_ref[i,i] = 0
                latt = qib.lattice.CustomizedLattice(shape, adj_ref)
                for i in range(latt.nsites):
                    self.assertEqual(i, latt.coord_to_index(latt.index_to_coord(i)))

        # three-dimensional lattice
        for Lx in range(3, 5):
            for Ly in range(3, 5):
                for Lz in range(3, 5):
                    shape = (Lx, Ly, Lz)
                    nsites = math.prod(shape)
                    adj_ref = rng.integers(-10, 10, (nsites, nsites))
                    adj_ref += adj_ref.T
                    for i in range(nsites):
                        adj_ref[i,i] = 0
                    latt = qib.lattice.CustomizedLattice(shape, adj_ref)
                    for i in range(latt.nsites):
                        self.assertEqual(i, latt.coord_to_index(latt.index_to_coord(i)))


if __name__ == "__main__":
    unittest.main()
