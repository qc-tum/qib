import numpy as np
from scipy import sparse
import unittest
import qib


class TestMolecularHamiltonian(unittest.TestCase):

    def test_molecular_hamiltonian(self):
        """
        Test construction of a molecular Hamiltonian.
        """
        # underlying lattice
        latt = qib.lattice.FullyConnectedLattice((4,))
        field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
        # Hamiltonian coefficients
        c = np.random.standard_normal()
        tkin = np.random.standard_normal((4, 4))
        tkin = 0.5 * (tkin + tkin.conj().T)
        vint = np.random.standard_normal((4, 4, 4, 4))
        H = qib.operator.MolecularHamiltonian(field, c, tkin, vint)
        self.assertEqual(H.num_orbitals, 4)
        self.assertEqual(H.fields(), [field])
        # reference matrices
        Tkin = construct_one_body_operator(tkin)
        Vint = construct_two_body_operator(vint)
        # compare
        self.assertAlmostEqual(sparse.linalg.norm(H.as_matrix() - (c*sparse.identity(2**4) + Tkin + Vint)), 0)


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


def construct_one_body_operator(h):
    """
    Construct a one-body operator.
    """
    # number of lattice sites
    L = len(h)
    T = sparse.csr_matrix((2**L, 2**L), dtype=float)
    for i in range(L):
        for j in range(L):
            T += h[i, j] * (fermi_create_op(L, 1 << (L-i-1)) @ fermi_annihil_op(L, 1 << (L-j-1)))
    return T


def construct_two_body_operator(h):
    """
    Construct a two body operator.
    """
    # number of lattice sites
    L = len(h)
    T = sparse.csr_matrix((2**(L), 2**(L)), dtype=float)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for l in range(L):
                    T += h[i, j, k, l] * 0.5 * (  fermi_create_op (L, 1 << (L-i-1))
                                                @ fermi_create_op (L, 1 << (L-j-1))
                                                @ fermi_annihil_op(L, 1 << (L-k-1))
                                                @ fermi_annihil_op(L, 1 << (L-l-1)))
    return T


if __name__ == "__main__":
    unittest.main()
