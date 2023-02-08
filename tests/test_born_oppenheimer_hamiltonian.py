import numpy as np
from scipy import sparse
import unittest
import qib


class TestBornOppenheimerHamiltonian(unittest.TestCase):

    def test_born_oppenhemier(self):
        """
        Test construction of the Born-Oppenheimer Hamiltonian.
        """
        # underlying lattice
        latt = qib.lattice.FullyConnectedLattice((4,))
        field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
        # parameters
        # construct Hamiltonian 
        # Example of 4 spin-orbitals: molecule H2 with sto3g basis
        np.random.seed(0)
        h0 = np.random.rand()
        h1 = np.random.rand(4,4)
        h2 = np.random.rand(4,4,4,4)
        H = qib.operator.BornOppenheimerHamiltonian(field, h0, h1, h2)
        self.assertEqual(H.fields(), [field])
        self.assertTrue(H.is_hermitian())
        # reference matrices
        H1 = construct_one_body_op(h1)
        H2 = construct_two_body_op(h2)
        # compare
        self.assertTrue(np.allclose(H.as_matrix().toarray(), (H1+H2).toarray()))
        
        
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


def construct_one_body_op(h1):
    """
    Construct the one body term of the molecule.
    """
    # number of lattice sites
    L = len(h1)
    T = sparse.csr_matrix((2**(L), 2**(L)), dtype=float)
    for i in range(L):
        for j in range(L):
            T += h1[i, j] * (fermi_create_op(L, 1 << (L-i-1)) @ fermi_annihil_op(L, 1 << (L-j-1)))
    return T


def construct_two_body_op(h2):
    """
    Construct the two body term of the molecule.
    """
    # number of lattice sites
    L = len(h2)
    T = sparse.csr_matrix((2**(L), 2**(L)), dtype=float)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for l in range(L):
                    T += h2[i, j, k, l] * (fermi_create_op(L, 1 << (L-i-1)) @ (fermi_annihil_op(L, 1 << (L-j-1)) @ fermi_create_op(L, 1 << (L-k-1))) @ fermi_annihil_op(L, 1 << (L-l-1)))
    return T


if __name__ == "__main__":
    unittest.main()
