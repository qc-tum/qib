import numpy as np
from scipy import sparse
import unittest
import qib


class TestFieldOperator(unittest.TestCase):

    def test_fermi_field_operator(self):
        """
        Test fermionic field operator functionality.
        """
        for pbc in [False, True]:
            # construct fermionic reference Hamiltonian
            lattsize = (3, 4)
            # onsite energy coefficients
            μ = np.random.random_sample(lattsize)
            # hopping and superconducting pairing coefficients
            # (last dimension corresponds to hopping in x- or y-direction)
            t = np.random.random_sample(lattsize + (2,))
            Δ = np.random.random_sample(lattsize + (2,))
            # reference Hamiltonian
            Href = construct_tight_binding_hamiltonian(μ, t, Δ, pbc=pbc)
            # must be symmetric
            self.assertEqual(sparse.linalg.norm(Href - Href.conj().T), 0)

            # construct fermionic field operator
            latt = qib.lattice.IntegerLattice(lattsize, pbc=pbc)
            adj_x = latt.adjacency_matrix_axis_shift(0, -1)
            adj_y = latt.adjacency_matrix_axis_shift(1, -1)
            field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
            # onsite term
            onsite_term = qib.operator.FieldOperatorTerm(
                [qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_CREATE),
                 qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_ANNIHIL)],
                np.diag(-μ.reshape(-1)))
            self.assertTrue(onsite_term.is_hermitian())
            # kinetic term
            tcoeffs = -(np.diag(t[:, :, 0].reshape(-1)) @ adj_x
                      + np.diag(t[:, :, 1].reshape(-1)) @ adj_y)
            tcoeffs = tcoeffs + tcoeffs.T
            kinetic_term = qib.operator.FieldOperatorTerm(
                [qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_CREATE),
                 qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_ANNIHIL)],
                tcoeffs)
            self.assertTrue(kinetic_term.is_hermitian())
            # superconducting pairing term
            Δcoeffs =  (np.diag(Δ[:, :, 0].reshape(-1)) @ adj_x
                      + np.diag(Δ[:, :, 1].reshape(-1)) @ adj_y)
            Δcoeffs = [Δcoeffs, Δcoeffs.T]
            sc_terms = [
                qib.operator.FieldOperatorTerm(
                    [qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_ANNIHIL),
                     qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_ANNIHIL)],
                    Δcoeffs[0]),
                qib.operator.FieldOperatorTerm(
                    [qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_CREATE),
                     qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_CREATE)],
                    Δcoeffs[1])]
            self.assertFalse(sc_terms[0].is_hermitian())
            self.assertFalse(sc_terms[1].is_hermitian())
            H = qib.FieldOperator([onsite_term, kinetic_term, sc_terms[0], sc_terms[1]])
            # compare
            self.assertAlmostEqual(sparse.linalg.norm(H.as_matrix() - Href), 0)



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


def construct_tight_binding_hamiltonian(μ: np.ndarray, t: np.ndarray, Δ: np.ndarray, pbc=False):
    """
    Construct a fermionic tight-binding Hamiltonian on a two-dimensional Cartesian lattice
    with open boundary conditions and per-site coefficients.
    """
    assert μ.ndim == 2
    lattsize = μ.shape
    nmodes = lattsize[0] * lattsize[1]
    H = sparse.csr_matrix((2**nmodes, 2**nmodes), dtype=float)
    # onsite term
    for x in range(lattsize[0]):
        for y in range(lattsize[1]):
            i = x*lattsize[1] + y
            H -= μ[x, y] * fermi_number_op(nmodes, 1 << i)
    for x in range(lattsize[0] if pbc else lattsize[0] - 1):
        x_next = (x + 1) % lattsize[0]
        for y in range(lattsize[1]):
            i = x     *lattsize[1] + y
            j = x_next*lattsize[1] + y
            # kinetic hopping in x-direction
            H -= t[x, y, 0] * (fermi_create_op(nmodes, 1 << j) @ fermi_annihil_op(nmodes, 1 << i) +
                               fermi_create_op(nmodes, 1 << i) @ fermi_annihil_op(nmodes, 1 << j))
            # superconducting pairing in x-direction
            H += Δ[x, y, 0] * (fermi_annihil_op(nmodes, 1 << i) @ fermi_annihil_op(nmodes, 1 << j) +
                                fermi_create_op(nmodes, 1 << j) @  fermi_create_op(nmodes, 1 << i))
    for x in range(lattsize[0]):
        for y in range(lattsize[1] if pbc else lattsize[1] - 1):
            y_next = (y + 1) % lattsize[1]
            i = x*lattsize[1] + y
            j = x*lattsize[1] + y_next
            # kinetic hopping in y-direction
            H -= t[x, y, 1] * (fermi_create_op(nmodes, 1 << j) @ fermi_annihil_op(nmodes, 1 << i) +
                               fermi_create_op(nmodes, 1 << i) @ fermi_annihil_op(nmodes, 1 << j))
            # superconducting pairing in y-direction
            H += Δ[x, y, 1] * (fermi_annihil_op(nmodes, 1 << i) @ fermi_annihil_op(nmodes, 1 << j) +
                                fermi_create_op(nmodes, 1 << j) @  fermi_create_op(nmodes, 1 << i))
    H.eliminate_zeros()
    return H


if __name__ == "__main__":
    unittest.main()
