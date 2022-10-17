import numpy as np
import unittest
import qib


class TestSpinLattice(unittest.TestCase):

    def test_integer_lattice_adjacency(self):
        """
        Test construction of adjacency matrices for spin-integer lattices.
        """
        for s in range(1,4):
            for L in range(3, 10):
                connect = np.identity(L, dtype=int)
                # one-dimensional lattice, periodic boundary conditions
                latt = qib.lattice.IntegerLattice((L,), pbc=True)
                spin_latt = qib.lattice.SpinLattice(latt,s)
                adj = spin_latt.adjacency_matrix()
                self.assertTrue(np.array_equal(adj, adj.T))
                adj_ref_original = np.roll(np.identity(L), 1, axis=0) + np.roll(np.identity(L), -1, axis=0)
                adj_ref = np.block([[connect]*i + [adj_ref_original] + [connect]*(s-i-1) for i in range(s)])
                self.assertEqual(latt.shape + (s,), spin_latt.shape)
                self.assertTrue(s*latt.nsites, spin_latt.nsites)
                self.assertTrue(np.array_equal(adj, adj_ref))

                # one-dimensional lattice, open boundary conditions
                latt = qib.lattice.IntegerLattice((L,), pbc=False)
                spin_latt = qib.lattice.SpinLattice(latt,s)
                adj = spin_latt.adjacency_matrix()
                self.assertTrue(np.array_equal(adj, adj.T))
                adj_ref_original = np.diag(np.ones(L - 1, dtype=int), k=1) + np.diag(np.ones(L - 1, dtype=int), k=-1)
                adj_ref = np.block([[connect]*i + [adj_ref_original] + [connect]*(s-i-1) for i in range(s)])
                self.assertTrue(np.array_equal(adj, adj_ref))

            for Lx in range(2, 5):
                for Ly in range(2, 5):
                    connect = np.identity(Lx*Ly, dtype=int)
                    # two-dimensional lattice, periodic boundary conditions
                    latt = qib.lattice.IntegerLattice((Lx, Ly), pbc=True)
                    spin_latt = qib.lattice.SpinLattice(latt,s)
                    adj = spin_latt.adjacency_matrix()
                    self.assertTrue(np.array_equal(adj, adj.T))
                    enum_latt = [(x, y) for x in range(Lx) for y in range(Ly)]
                    adj_ref_original = np.zeros((len(enum_latt), len(enum_latt)), dtype=int)
                    for i, a in enumerate(enum_latt):
                        for j, b in enumerate(enum_latt):
                            if abs((a[0] - b[0] + 1) % Lx - 1) + abs((a[1] - b[1] + 1) % Ly - 1) == 1:
                                adj_ref_original[i, j] = 1
                    adj_ref = np.block([[connect]*i + [adj_ref_original] + [connect]*(s-i-1) for i in range(s)])
                    self.assertTrue(np.array_equal(adj, adj_ref))

                    # two-dimensional lattice, open boundary conditions
                    latt = qib.lattice.IntegerLattice((Lx, Ly), pbc=False)
                    spin_latt = qib.lattice.SpinLattice(latt,s)
                    adj = spin_latt.adjacency_matrix()
                    self.assertTrue(np.array_equal(adj, adj.T))
                    enum_latt = [(x, y) for x in range(Lx) for y in range(Ly)]
                    adj_ref_original = np.zeros((len(enum_latt), len(enum_latt)), dtype=int)
                    for i, a in enumerate(enum_latt):
                        for j, b in enumerate(enum_latt):
                            if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                                adj_ref_original[i, j] = 1
                    adj_ref = np.block([[connect]*i + [adj_ref_original] + [connect]*(s-i-1) for i in range(s)])
                    self.assertTrue(np.array_equal(adj, adj_ref))

                    # two-dimensional lattice, open boundary conditions in
                    # x-direction and periodic boundary conditions in y-direction
                    latt = qib.lattice.IntegerLattice((Lx, Ly), pbc=(False, True))
                    spin_latt = qib.lattice.SpinLattice(latt,s)
                    adj = spin_latt.adjacency_matrix()
                    self.assertTrue(np.array_equal(adj, adj.T))
                    enum_latt = [(x, y) for x in range(Lx) for y in range(Ly)]
                    adj_ref_original = np.zeros((len(enum_latt), len(enum_latt)), dtype=int)
                    for i, a in enumerate(enum_latt):
                        for j, b in enumerate(enum_latt):
                            if abs(a[0] - b[0]) + abs((a[1] - b[1] + 1) % Ly - 1) == 1:
                                adj_ref_original[i, j] = 1
                    adj_ref = np.block([[connect]*i + [adj_ref_original] + [connect]*(s-i-1) for i in range(s)])
                    self.assertTrue(np.array_equal(adj, adj_ref))

    def test_hexagonal_lattice_adjacency_with_delete(self):
        """
        Test construction of adjacency matrices for spin-hexagonal lattices.
        Delete option enabled.
        """
        for s in range(1,4):
            for Lx in range(1, 5):
                for Ly in range(1, 5):
                    # two-dimensional lattice, open boundary conditions, COLS_SHIFTED_UP convention
                    latt = qib.lattice.HexagonalLattice((Lx, Ly), pbc=False, delete=True, convention=qib.lattice.HexagonalLatticeConvention.COLS_SHIFTED_UP)
                    connect = np.identity(latt.nsites, dtype=int)
                    spin_latt = qib.lattice.SpinLattice(latt,s)
                    adj = spin_latt.adjacency_matrix()
                    self.assertTrue(np.array_equal(adj, adj.T))
                    if Ly>1:
                        nrows_square = 2*Lx+2
                    else:
                        nrows_square = 2*Lx+1
                    ncols_square = Ly+1
                    enum_latt = [(x, y) for x in range(nrows_square) for y in range(ncols_square)]
                    adj_ref_original = np.zeros((len(enum_latt), len(enum_latt)), dtype=int)
                    for i, a in enumerate(enum_latt):
                        for j, b in enumerate(enum_latt):
                            if a[1] == b[1] and abs(a[0]-b[0]) == 1:
                                adj_ref_original[i, j] = 1
                            elif a[0] == b[0] and abs(a[1]-b[1])==1:
                                if (max(a[1],b[1])+a[0])%2 == 1:
                                    adj_ref_original[i,j] = 1
                    
                    # delete extra points
                    if Ly > 1:
                        adj_ref_original = np.delete(adj_ref_original, (2*Lx+1)*(Ly+1), 0)
                        adj_ref_original = np.delete(adj_ref_original, (2*Lx+1)*(Ly+1), 1)
                        if Ly%2 == 1:
                            adj_ref_original= np.delete(adj_ref_original, -1, 0)
                            adj_ref_original= np.delete(adj_ref_original, -1, 1)
                        else:
                            adj_ref_original = np.delete(adj_ref_original, Ly, 0)
                            adj_ref_original = np.delete(adj_ref_original, Ly, 1)
                    adj_ref = np.block([[connect]*i + [adj_ref_original] + [connect]*(s-i-1) for i in range(s)])
                    self.assertEqual(latt.shape + (s,), spin_latt.shape)
                    self.assertTrue(s*latt.nsites, spin_latt.nsites)
                    self.assertTrue(np.array_equal(adj, adj_ref))

                    # two-dimensional lattice, open boundary conditions ROWS_SHIFTED_LEFT convention
                    latt = qib.lattice.HexagonalLattice((Lx, Ly), pbc=False, delete=True, convention=qib.lattice.HexagonalLatticeConvention.ROWS_SHIFTED_LEFT)
                    connect = np.identity(latt.nsites, dtype=int)
                    spin_latt = qib.lattice.SpinLattice(latt,s)
                    adj = spin_latt.adjacency_matrix()
                    self.assertTrue(np.array_equal(adj, adj.T))
                    if Lx>1:
                        ncols_square = 2*Ly+2
                    else:
                        ncols_square = 2*Ly+1
                    nrows_square = Lx+1
                    enum_latt = [(x, y) for x in range(nrows_square) for y in range(ncols_square)]
                    adj_ref_original = np.zeros((len(enum_latt), len(enum_latt)), dtype=int)
                    for i, a in enumerate(enum_latt):
                        for j, b in enumerate(enum_latt):
                            if a[0] == b[0] and abs(a[1]-b[1]) == 1:
                                adj_ref_original[i, j] = 1
                            elif a[1] == b[1] and abs(a[0]-b[0])==1:
                                if (max(a[0],b[0])+a[1])%2 == 1 :
                                    adj_ref_original[i,j] = 1
                    # delete extra points        
                    if Lx > 1:
                        adj_ref_original = np.delete(adj_ref_original, 2*Ly+1, 0)
                        adj_ref_original = np.delete(adj_ref_original, 2*Ly+1, 1)
                        if Lx%2 == 1:
                            adj_ref_original = np.delete(adj_ref_original, -1, 0)
                            adj_ref_original = np.delete(adj_ref_original, -1, 1)
                        else:
                            adj_ref_original = np.delete(adj_ref_original, Lx*(2*Ly+2)-1, 0)
                            adj_ref_original = np.delete(adj_ref_original, Lx*(2*Ly+2)-1, 1)
                    adj_ref = np.block([[connect]*i + [adj_ref_original] + [connect]*(s-i-1) for i in range(s)])
                    self.assertEqual(latt.shape + (s,), spin_latt.shape)
                    self.assertTrue(s*latt.nsites, spin_latt.nsites)
                    self.assertTrue(np.array_equal(adj, adj_ref))


    def test_lattice_adjacency_without_delete(self):
        """
        Test construction of adjacency matrices for spin-hexagonal lattices.
        Delete option disabled.
        """
        for s in range(1,4):
            for Lx in range(1, 5):
                for Ly in range(1, 5):
                    # two-dimensional lattice, open boundary conditions, COLS_SHIFTED_UP convention
                    latt = qib.lattice.HexagonalLattice((Lx, Ly), pbc=False, delete=False, convention=qib.lattice.HexagonalLatticeConvention.COLS_SHIFTED_UP)
                    connect = np.identity(latt.nsites, dtype=int)
                    spin_latt = qib.lattice.SpinLattice(latt,s)
                    adj = spin_latt.adjacency_matrix()
                    self.assertTrue(np.array_equal(adj, adj.T))
                    if Ly>1:
                        nrows_square = 2*Lx+2
                    else:
                        nrows_square = 2*Lx+1
                    ncols_square = Ly+1
                    enum_latt = [(x, y) for x in range(nrows_square) for y in range(ncols_square)]
                    adj_ref_original = np.zeros((len(enum_latt), len(enum_latt)), dtype=int)
                    for i, a in enumerate(enum_latt):
                        for j, b in enumerate(enum_latt):
                            if a[1] == b[1] and abs(a[0]-b[0]) == 1:
                                adj_ref_original[i, j] = 1
                            elif a[0] == b[0] and abs(a[1]-b[1])==1:
                                if (max(a[1],b[1])+a[0])%2 == 1:
                                    adj_ref_original[i,j] = 1
                    # disconnect extra points
                    if Ly > 1:
                        adj_ref_original[(2*Lx+1)*(Ly+1), :] = 0
                        adj_ref_original[:, (2*Lx+1)*(Ly+1)] = 0
                        if Ly%2 == 1:
                            adj_ref_original[-1, :] = 0
                            adj_ref_original[:, -1] = 0
                        else:
                            adj_ref_original[Ly, :] = 0
                            adj_ref_original[:, Ly] = 0
                    adj_ref = np.block([[connect]*i + [adj_ref_original] + [connect]*(s-i-1) for i in range(s)])
                    self.assertEqual(latt.shape + (s,), spin_latt.shape)
                    self.assertTrue(s*latt.nsites, spin_latt.nsites)
                    self.assertTrue(np.array_equal(adj, adj_ref))

                    # two-dimensional lattice, open boundary conditions ROWS_SHIFTED_LEFT convention
                    latt = qib.lattice.HexagonalLattice((Lx, Ly), pbc=False, convention=qib.lattice.HexagonalLatticeConvention.ROWS_SHIFTED_LEFT)
                    connect = np.identity(latt.nsites, dtype=int)
                    spin_latt = qib.lattice.SpinLattice(latt,s)
                    adj = spin_latt.adjacency_matrix()
                    self.assertTrue(np.array_equal(adj, adj.T))
                    if Lx>1:
                        ncols_square = 2*Ly+2
                    else:
                        ncols_square = 2*Ly+1
                    nrows_square = Lx+1
                    enum_latt = [(x, y) for x in range(nrows_square) for y in range(ncols_square)]
                    adj_ref_original = np.zeros((len(enum_latt), len(enum_latt)), dtype=int)
                    for i, a in enumerate(enum_latt):
                        for j, b in enumerate(enum_latt):
                            if a[0] == b[0] and abs(a[1]-b[1]) == 1:
                                adj_ref_original[i, j] = 1
                            elif a[1] == b[1] and abs(a[0]-b[0])==1:
                                if (max(a[0],b[0])+a[1])%2 == 1 :
                                    adj_ref_original[i,j] = 1
                    # disconnect extra points        
                    if Lx > 1:
                        adj_ref_original[2*Ly+1, :] = 0
                        adj_ref_original[:, 2*Ly+1] = 0
                        if Lx%2 == 1:
                            adj_ref_original[-1, :] = 0
                            adj_ref_original[:, -1] = 0
                        else:
                            adj_ref_original[Lx*(2*Ly+2), :] = 0
                            adj_ref_original[:, Lx*(2*Ly+2)] = 0
                    adj_ref = np.block([[connect]*i + [adj_ref_original] + [connect]*(s-i-1) for i in range(s)])
                    self.assertEqual(latt.shape + (s,), spin_latt.shape)
                    self.assertTrue(s*latt.nsites, spin_latt.nsites)
                    self.assertTrue(np.array_equal(adj, adj_ref))

    def test_integer_lattice_coords(self):
        """
        Test lattice coordinate indexing.
        """
        for s in range(1,4):
            # one-dimensional lattice
            for L in range(3, 10):
                latt = qib.lattice.IntegerLattice((L,))
                spin_latt = qib.lattice.SpinLattice(latt,s)
                for i in range(spin_latt.nsites):
                    self.assertEqual(i, spin_latt.coord_to_index(spin_latt.index_to_coord(i)))

            # two-dimensional lattice
            for Lx in range(3, 5):
                for Ly in range(3, 5):
                    latt = qib.lattice.IntegerLattice((Lx, Ly))
                    spin_latt = qib.lattice.SpinLattice(latt,s)
                    for i in range(spin_latt.nsites):
                        self.assertEqual(i, spin_latt.coord_to_index(spin_latt.index_to_coord(i)))

            # three-dimensional lattice
            for Lx in range(3, 5):
                for Ly in range(3, 5):
                    for Lz in range(3, 5):
                        latt = qib.lattice.IntegerLattice((Lx, Ly, Lz))
                        spin_latt = qib.lattice.SpinLattice(latt,s)
                        for i in range(spin_latt.nsites):
                            self.assertEqual(i, spin_latt.coord_to_index(spin_latt.index_to_coord(i)))

    def test_hexagonal_lattice_coords(self):
        """
        Test lattice coordinate indexing.
        """
        # two-dimensional lattice, open boundary conditions
        for s in range(1,4):
            for Lx in range(1, 5):
                for Ly in range(1, 5):
                    latt_1 = qib.lattice.SpinLattice(qib.lattice.HexagonalLattice((Lx, Ly), pbc=False, delete=False, convention=qib.lattice.HexagonalLatticeConvention.COLS_SHIFTED_UP), s)
                    latt_2 = qib.lattice.SpinLattice(qib.lattice.HexagonalLattice((Lx, Ly), pbc=False, delete=False, convention=qib.lattice.HexagonalLatticeConvention.ROWS_SHIFTED_LEFT), s)
                    for i in range(latt_1.nsites):
                        self.assertEqual(i, latt_1.coord_to_index(latt_1.index_to_coord(i)))
                    for i in range(latt_2.nsites):
                        self.assertEqual(i, latt_2.coord_to_index(latt_2.index_to_coord(i)))
                    latt_1 = qib.lattice.SpinLattice(qib.lattice.HexagonalLattice((Lx, Ly), pbc=False, delete=True, convention=qib.lattice.HexagonalLatticeConvention.COLS_SHIFTED_UP), s)
                    latt_2 = qib.lattice.SpinLattice(qib.lattice.HexagonalLattice((Lx, Ly), pbc=False, delete=True, convention=qib.lattice.HexagonalLatticeConvention.ROWS_SHIFTED_LEFT), s)
                    for i in range(latt_1.nsites):
                        self.assertEqual(i, latt_1.coord_to_index(latt_1.index_to_coord(i)))
                    for i in range(latt_2.nsites):
                        self.assertEqual(i, latt_2.coord_to_index(latt_2.index_to_coord(i)))

if __name__ == "__main__":
    unittest.main()
