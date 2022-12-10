import numpy as np
from scipy import sparse
import unittest
import qib


class TestFermiHubbardHamiltonian(unittest.TestCase):

    def test_spinless_construction(self):
        """
        Test construction of the spinless Fermi-Hubbard Hamiltonian.
        """
        # underlying lattice
        latt = qib.lattice.HexagonalLattice((1, 2), pbc=False)
        field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
        # parameters
        t = np.random.uniform(-1, 1)
        u = np.random.uniform(0, 4)
        # construct Hamiltonian
        H = qib.operator.FermiHubbardHamiltonian(field, t, u, spin=False)
        self.assertEqual(H.fields(), [field])
        self.assertTrue(H.is_hermitian())
        # reference matrices
        adj = latt.adjacency_matrix()
        Tmat = construct_spinless_kinetic_matrix(-t * adj)
        Vmat = construct_neighbor_interation_matrix(adj, u)
        # compare
        self.assertAlmostEqual(sparse.linalg.norm(H.as_matrix() - (Tmat + Vmat)), 0)

    def test_spinful_construction(self):
        """
        Test construction of the spinful Fermi-Hubbard Hamiltonian.
        """
        # underlying spinful lattice
        latt = qib.lattice.IntegerLattice((2, 3), pbc=False)
        field = qib.field.Field(qib.field.ParticleType.FERMION, qib.lattice.LayeredLattice(latt, 2))
        # parameters
        t = np.random.uniform(-1, 1)
        u = np.random.uniform(0, 4)
        # construct Hamiltonian
        H = qib.operator.FermiHubbardHamiltonian(field, t, u, spin=True)
        self.assertEqual(H.fields(), [field])
        self.assertTrue(H.is_hermitian())
        # reference matrices
        Tmat = construct_spinful_kinetic_matrix(-t * latt.adjacency_matrix())
        Vmat = construct_spin_interation_matrix(latt.nsites, u)
        # compare
        self.assertAlmostEqual(sparse.linalg.norm(H.as_matrix() - (Tmat + Vmat)), 0)



# Alternative implementation of fermionic operators, as reference


def fermi_annihil_sign(n, a):
    """
    Sign factor of annihilating modes encoded in `a` as 1-bits
    applied to state with occupied modes represented by `n`.
    """
    if n & a == a:
        na = n - a
        counter = 0
        while a:
            # current least significant bit
            lsb = (a & -a)
            counter += (na & (lsb - 1)).bit_count()
            a -= lsb
        return 1 - 2*(counter % 2)
    else:
        # applying annihilation operator yields zero
        return 0


def fermi_create_sign(n, c):
    """
    Sign factor of creating modes encoded in `c` as 1-bits
    applied to state with occupied modes represented by `n`.
    """
    if n & c == 0:
        counter = 0
        while c:
            # current least significant bit
            lsb = (c & -c)
            counter += (n & (lsb - 1)).bit_count()
            c -= lsb
        return 1 - 2*(counter % 2)
    else:
        # applying creation operator yields zero
        return 0


def fermi_annihil_op(nmodes, a):
    """
    Fermionic annihilation operator on full Fock space.
    """
    data = np.array([fermi_annihil_sign(n, a) for n in range(2**nmodes)], dtype=float)
    row_ind = np.arange(2**nmodes) - a
    col_ind = np.arange(2**nmodes)
    nzi = np.nonzero(data)[0]
    return sparse.csr_matrix((data[nzi], (row_ind[nzi], col_ind[nzi])), shape=(2**nmodes, 2**nmodes))


def fermi_create_op(nmodes, c):
    """
    Fermionic creation operator on full Fock space.
    """
    data = np.array([fermi_create_sign(n, c) for n in range(2**nmodes)], dtype=float)
    row_ind = np.arange(2**nmodes) + c
    col_ind = np.arange(2**nmodes)
    nzi = np.nonzero(data)[0]
    return sparse.csr_matrix((data[nzi], (row_ind[nzi], col_ind[nzi])), shape=(2**nmodes, 2**nmodes))


def fermi_number_op(nmodes, f):
    """
    Fermionic number operator on full Fock space.
    """
    data = np.array([1 if (n & f == f) else 0 for n in range(2**nmodes)], dtype=float)
    ind = np.arange(2**nmodes)
    nzi = np.nonzero(data)[0]
    return sparse.csr_matrix((data[nzi], (ind[nzi], ind[nzi])), shape=(2**nmodes, 2**nmodes))


def construct_spinless_kinetic_matrix(coeffs: np.ndarray):
    """
    Construct the spinless kinetic hopping term as sparse matrix.
    """
    assert coeffs.ndim == 2
    assert coeffs.shape[0] == coeffs.shape[1]
    # number of lattice sites
    L = coeffs.shape[0]
    T = sparse.csr_matrix((2**L, 2**L), dtype=float)
    for i in range(L):
        for j in range(i + 1, L):
            if coeffs[i, j] == 0:
                continue
            T += coeffs[i, j] * (fermi_create_op(L, 1 << j) @ fermi_annihil_op(L, 1 << i) +
                                 fermi_create_op(L, 1 << i) @ fermi_annihil_op(L, 1 << j))
    return T


def construct_spinful_kinetic_matrix(coeffs: np.ndarray):
    """
    Construct the spinful kinetic hopping term as sparse matrix.
    """
    assert coeffs.ndim == 2
    assert coeffs.shape[0] == coeffs.shape[1]
    # number of spinful lattice sites
    L = coeffs.shape[0]
    T = sparse.csr_matrix((2**(2*L), 2**(2*L)), dtype=float)
    for i in range(L):
        for j in range(i + 1, L):
            if coeffs[i, j] == 0:
                continue
            # sum over spin
            for s in range(2):
                T += coeffs[i, j] * (fermi_create_op(2*L, 1 << (j + s*L)) @ fermi_annihil_op(2*L, 1 << (i + s*L)) +
                                     fermi_create_op(2*L, 1 << (i + s*L)) @ fermi_annihil_op(2*L, 1 << (j + s*L)))
    return T


def construct_neighbor_interation_matrix(adj: np.ndarray, u: float):
    """
    Construct the sparse matrix representation of the neighbor interaction term
    of the spinless Fermi-Hubbard model.
    """
    # number of lattice sites
    L = adj.shape[0]
    V = sparse.csr_matrix((2**L, 2**L), dtype=float)
    for i in range(L):
        for j in range(i + 1, L):
            if adj[i, j] != 0:
                V += u * (fermi_number_op(L, 1 << j) @ fermi_number_op(L, 1 << i))
    return V


def construct_spin_interation_matrix(L: int, u: float):
    """
    Construct the sparse matrix representation of the spin interaction term
    of the Fermi-Hubbard model.
    """
    V = sparse.csr_matrix((2**(2*L), 2**(2*L)), dtype=float)
    for i in range(L):
        V += u * (fermi_number_op(2*L, 1 << (i + L)) @ fermi_number_op(2*L, 1 << i))
    return V


if __name__ == "__main__":
    unittest.main()
