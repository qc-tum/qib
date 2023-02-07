import numpy as np
from scipy import sparse
from pyscf import gto, scf, ao2mo
import unittest
import sys
sys.path.append('../src')
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
        atom = '''H 0 0 0; H 0 0 0.735'''
        basis = 'sto-6g'
        charge = 0
        spin = 0
        verbose = 0
        mol = gto.Mole()
        mol.build(atom = atom,
                  basis = basis,
                  charge = charge,
                  spin = spin,
                  verbose = verbose)
        # construct Hamiltonian (SPIN-ORBITAL basis!)
        h0 = mol.get_enuc()
        h1 = np.kron(mol.get_hcore(), np.identity(2))
        h2 = mol.intor('int2e_spinor')
        H = qib.operator.BornOppenheimerHamiltonian(field, h0, h1, h2)
        self.assertEqual(H.fields(), [field])
        self.assertTrue(H.is_hermitian())
        
        # reference matrices
        #H0 = mol.energy_nuc()
        H1 = construct_one_body_op(mol)
        H2 = construct_two_body_op(mol)
        # compare
        self.assertAlmostEqual(sparse.linalg.norm(H.as_matrix() - (H1+H2)), 0)
        # Hartree Fock
        myhf = scf.ROHF(mol)
        myhf.kernel()
        # coefficient matrix
        coeff = myhf.mo_coeff
        # MO basis integrals
        #n_spo = 2*mol.nao_nr()        
        mo_H1 = construct_one_body_op_mobasis(mol, coeff)
        mo_H2 = construct_two_body_op_mobasis(mol, coeff)
        H.set_mo_integrals(coeff)
        #print(sparse.linalg.norm(mo_H1+mo_H2))
        # compare
        self.assertAlmostEqual(sparse.linalg.norm(H.as_matrix()), sparse.linalg.norm(mo_H1+mo_H2))
        #BUG: norm changes? coeff is not unitary!
        

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


def construct_one_body_op(mol):
    """
    Construct the one body term of the molecule.
    """
    # number of lattice sites
    L = mol.nao_nr()
    T = sparse.csr_matrix((2**(2*L), 2**(2*L)), dtype=float)
    h1 = np.kron(mol.get_hcore(), np.identity(2))
    
    for i in range(2*L):
        for j in range(2*L):
            T += h1[i, j] * (fermi_create_op(2*L, 1 << i) @ fermi_annihil_op(2*L, 1 << j))
    return T

def construct_one_body_op_mobasis(mol, coeff):
    """
    Construct the one body term of the molecule (molecular orbitals' basis').
    """
    # number of lattice sites
    L = mol.nao_nr()
    T = sparse.csr_matrix((2**(2*L), 2**(2*L)), dtype=float)
    spin_coeff = np.kron(coeff, np.identity(2))
    h1 = np.einsum('ji,jk,kl->il', spin_coeff, np.kron(mol.get_hcore(), np.identity(2)), spin_coeff)
    
    for i in range(2*L):
        for j in range(2*L):
            T += h1[i, j] * (fermi_create_op(2*L, 1 << i) @ fermi_annihil_op(2*L, 1 << j))
    return T

def construct_two_body_op(mol):
    """
    Construct the two body term of the molecule.
    """
    # number of lattice sites
    L = mol.nao_nr()
    T = sparse.csr_matrix((2**(2*L), 2**(2*L)), dtype=float)
    h2= mol.intor('int2e_spinor')
    for i in range(2*L):
        for j in range(2*L):
            for k in range(2*L):
                for l in range(2*L):
                    T += h2[i, j, k, l] * (fermi_create_op(2*L, 1 << i) @ (fermi_annihil_op(2*L, 1 << j) @ fermi_create_op(2*L, 1 << k)) @ fermi_annihil_op(2*L, 1 << l))
    return T

def construct_two_body_op_mobasis(mol, coeff):
    """
    Construct the two body term of the molecule (molecular orbitals' basis')
    """
    # number of lattice sites
    L = mol.nao_nr()
    T = sparse.csr_matrix((2**(2*L), 2**(2*L)), dtype=float)
    spin_coeff = np.kron(coeff, np.identity(2))
    h2= ao2mo.kernel(mol.intor('int2e_spinor'), spin_coeff, compact=False)
    for i in range(2*L):
        for j in range(2*L):
            for k in range(2*L):
                for l in range(2*L):
                    T += h2[i, j, k, l] * (fermi_create_op(2*L, 1 << i) @ (fermi_annihil_op(2*L, 1 << j) @ fermi_create_op(2*L, 1 << k)) @ fermi_annihil_op(2*L, 1 << l))
    return T


if __name__ == "__main__":
    unittest.main()
