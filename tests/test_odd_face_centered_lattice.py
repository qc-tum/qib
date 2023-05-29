import unittest
import numpy as np
import qib


class TestOddFaceCenteredLattice(unittest.TestCase):

    def test_lattice_adjacency(self):
        """
        Test construction of adjacency matrices.
        """
        # only two-dimensional lattices supported
        for Lx in range(3, 7):
            for Ly in range(3, 7):
                latt = qib.lattice.OddFaceCenteredLattice((Lx, Ly))
                adj = latt.adjacency_matrix()
                for i in range(latt.nsites):
                    for j in range(latt.nsites):
                        dist = np.linalg.norm(
                            np.array(latt.index_to_coord(i)) -
                            np.array(latt.index_to_coord(j)))
                        # whether sites are neighbors based on distance
                        neigh_dist = (dist == 1 or abs(dist - 1./np.sqrt(2)) < 1e-14)
                        self.assertTrue(neigh_dist == adj[i, j])

    def test_lattice_coords(self):
        """
        Test lattice coordinate indexing.
        """
        # only two-dimensional lattices supported
        for Lx in range(3, 7):
            for Ly in range(3, 7):
                latt = qib.lattice.OddFaceCenteredLattice((Lx, Ly))
                for i in range(latt.nsites):
                    self.assertEqual(i, latt.coord_to_index(latt.index_to_coord(i)))

    def test_odd_face_adjacency(self):
        """
        Test finding of odd faces adjacent to a given edge.
        """
        # only two-dimensional lattices supported
        for Lx in range(3, 7):
            for Ly in range(3, 7):
                latt = qib.lattice.OddFaceCenteredLattice((Lx, Ly))
                # enumerate all site coordinates
                coords = [latt.index_to_coord(i) for i in range(latt.nsites)]
                # enumerate edges
                for x in range(latt.shape[0]):
                    for y in range(latt.shape[1]):
                        i = (x, y)
                        for d in ["horz", "vert"]:
                            if d == "horz":
                                if x + 1 < latt.shape[0]:
                                    j = (x + 1, y)
                                else:
                                    continue
                            else:
                                if y + 1 < latt.shape[1]:
                                    j = (x, y + 1)
                                else:
                                    continue
                            # search for adjacent odd face based on coordinates
                            iof_ref = -1
                            for k, c in enumerate(coords):
                                if (abs(np.linalg.norm(np.array(i) - np.array(c)) - 1./np.sqrt(2)) < 1e-14 and
                                    abs(np.linalg.norm(np.array(j) - np.array(c)) - 1./np.sqrt(2)) < 1e-14):
                                    iof_ref = k
                                    break
                            iof = latt.edge_to_odd_face_index(i, j)
                            self.assertEqual(iof, iof_ref)


if __name__ == "__main__":
    unittest.main()
