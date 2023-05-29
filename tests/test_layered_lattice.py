import unittest
import numpy as np
import qib


class TestLayeredLattice(unittest.TestCase):

    def test_lattice_adjacency(self):
        """
        Test construction of adjacency matrices.
        """
        # compare with adjacency matrix of an integer lattice in one higher dimension
        nlayers = 2
        # 1D base lattice
        latt = qib.lattice.LayeredLattice(qib.lattice.IntegerLattice((6,), pbc=True), nlayers)
        latt_ref = qib.lattice.IntegerLattice((nlayers, 6), pbc=True)
        self.assertEqual(latt.ndim, latt_ref.ndim)
        self.assertTrue(np.array_equal(latt.adjacency_matrix(), latt_ref.adjacency_matrix()))
        # 2D base lattice
        latt = qib.lattice.LayeredLattice(qib.lattice.IntegerLattice((5, 11), pbc=True), nlayers)
        latt_ref = qib.lattice.IntegerLattice((nlayers, 5, 11), pbc=True)
        self.assertEqual(latt.ndim, latt_ref.ndim)
        self.assertTrue(np.array_equal(latt.adjacency_matrix(), latt_ref.adjacency_matrix()))

        # construct reference adjacency matrix based on site coordinates
        nlayers = 5
        latt = qib.lattice.LayeredLattice(qib.lattice.IntegerLattice((4, 7), pbc=False), nlayers)
        self.assertEqual(latt.ndim, 3)
        adj_ref = np.zeros(2 * (latt.nsites,), dtype=int)
        for i in range(latt.nsites):
            c = latt.index_to_coord(i)
            for j in range(latt.nsites):
                if i == j:
                    continue
                d = latt.index_to_coord(j)
                if c[1:] == d[1:] or (c[0] == d[0] and np.linalg.norm(np.array(c[1:]) - np.array(d[1:])) == 1):
                    adj_ref[i, j] = 1
        # compare
        self.assertTrue(np.array_equal(latt.adjacency_matrix(), adj_ref))

    def test_lattice_coords(self):
        """
        Test lattice coordinate indexing.
        """
        for nlayers in range(1, 4):
            # two-dimensional integer base lattice
            latt = qib.lattice.LayeredLattice(qib.lattice.IntegerLattice((3, 4, 5)), nlayers)
            for i in range(latt.nsites):
                self.assertEqual(i, latt.coord_to_index(latt.index_to_coord(i)))
            # hexagonal base lattice
            for convention in qib.lattice.ShiftedLatticeConvention:
                hexlatt = qib.lattice.HexagonalLattice((3, 4), pbc=False, convention=convention)
                latt = qib.lattice.LayeredLattice(hexlatt, nlayers)
                for i in range(latt.nsites):
                    self.assertEqual(i, latt.coord_to_index(latt.index_to_coord(i)))


if __name__ == "__main__":
    unittest.main()
