import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
import unittest
import qib
import qib.operator as qop


class TestOperator(unittest.TestCase):

    def test_pauli_string(self):
        """
        Test handling of Pauli strings.
        """
        self.assertEqual(spla.norm(qop.PauliString.identity(5).as_matrix()
                                   - sparse.identity(2**5)), 0)
        # construct Pauli string
        P = qop.PauliString.from_single_paulis(5, ('Y', 1), ('X', 0), ('Y', 3), ('Z', 4), q=3)
        # reference values
        z_ref = [0, 1, 0, 1, 1]
        x_ref = [1, 1, 0, 1, 0]
        q_ref = 3
        self.assertTrue(np.array_equal(P.z, z_ref))
        self.assertTrue(np.array_equal(P.x, x_ref))
        self.assertEqual(P.q, q_ref)
        # reference matrix representation
        I = np.identity(2)
        X = np.array([[ 0.,  1.], [ 1.,  0.]])
        Y = np.array([[ 0., -1j], [ 1j,  0.]])
        Z = np.array([[ 1.,  0.], [ 0., -1.]])
        Pref = (-1j)**q_ref * np.kron(np.kron(np.kron(np.kron(X, Y), I), Y), Z)
        self.assertTrue(np.array_equal(P.as_matrix().toarray(), Pref))
        # another Pauli string
        P2 = qop.PauliString.from_single_paulis(5, ('Z', 4), ('Y', 0), ('Y', 1), ('X', 2), q=2)
        # logical product
        self.assertEqual(spla.norm(( P @ P2).as_matrix()
                                   - P.as_matrix() @ P2.as_matrix()), 0)
        # logical product for various lengths
        for nqubits in range(1, 10):
            Plist = []
            for j in range(2):
                z = np.random.randint(0, 2, nqubits)
                x = np.random.randint(0, 2, nqubits)
                q = np.random.randint(0, 4)
                Plist.append(qop.PauliString(z, x, q))
            self.assertEqual(spla.norm( (Plist[0] @ Plist[1]).as_matrix()
                                       - Plist[0].as_matrix() @ Plist[1].as_matrix()), 0)

    def test_pauli_operator(self):
        """
        Test Pauli operator functionality.
        """
        # construct Pauli strings
        z = [[0, 1, 0, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 0]]
        x = [[1, 1, 0, 1, 0], [0, 1, 1, 0, 0], [1, 0, 0, 0, 1]]
        q = [3, 0, 1]
        weights = [-1.52, 0.687, 0.135]
        P = qop.PauliOperator(
            [qop.WeightedPauliString(
                qop.PauliString(z[j], x[j], q[j]),
                weights[j]) for j in range(3)])
        # reference calculation
        I = np.identity(2)
        X = np.array([[ 0.,  1.], [ 1.,  0.]])
        Y = np.array([[ 0., -1j], [ 1j,  0.]])
        Z = np.array([[ 1.,  0.], [ 0., -1.]])
        Pref = (
              weights[0] * (-1j)**q[0] * np.kron(np.kron(np.kron(np.kron(X, Y), I), Y), Z)
            + weights[1] * (-1j)**q[1] * np.kron(np.kron(np.kron(np.kron(Z, X), Y), Z), Z)
            + weights[2] * (-1j)**q[2] * np.kron(np.kron(np.kron(np.kron(Y, Z), I), Z), X))
        # compare
        self.assertTrue(np.allclose(P.as_matrix().toarray(), Pref))

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
            self.assertEqual(spla.norm(Href - Href.conj().T), 0)

            # construct fermionic field operator
            latt = qib.lattice.IntegerLattice(lattsize, pbc=pbc)
            adj_x = latt.adjacency_matrix_axis_shift(0, -1)
            adj_y = latt.adjacency_matrix_axis_shift(1, -1)
            field = qop.Field(qop.ParticleType.FERMION, latt)
            # onsite term
            onsite_term = qop.FieldOperatorTerm(
                [qop.IFODesc(field, qop.IFOType.FERMI_CREATE),
                  qop.IFODesc(field, qop.IFOType.FERMI_ANNIHIL)],
                np.diag(-μ.reshape(-1)))
            # kinetic term
            tcoeffs = -(np.diag(t[:, :, 0].reshape(-1)) @ adj_x
                      + np.diag(t[:, :, 1].reshape(-1)) @ adj_y)
            tcoeffs = tcoeffs + tcoeffs.T
            kinetic_term = qop.FieldOperatorTerm(
                [qop.IFODesc(field, qop.IFOType.FERMI_CREATE),
                  qop.IFODesc(field, qop.IFOType.FERMI_ANNIHIL)],
                tcoeffs)
            # superconducting pairing term
            Δcoeffs =  (np.diag(Δ[:, :, 0].reshape(-1)) @ adj_x
                      + np.diag(Δ[:, :, 1].reshape(-1)) @ adj_y)
            Δcoeffs = [Δcoeffs, Δcoeffs.T]
            sc_terms = [
                qop.FieldOperatorTerm(
                    [qop.IFODesc(field, qop.IFOType.FERMI_ANNIHIL),
                      qop.IFODesc(field, qop.IFOType.FERMI_ANNIHIL)],
                    Δcoeffs[0]),
                qop.FieldOperatorTerm(
                    [qop.IFODesc(field, qop.IFOType.FERMI_CREATE),
                      qop.IFODesc(field, qop.IFOType.FERMI_CREATE)],
                    Δcoeffs[1])]
            H = qop.FieldOperator([onsite_term, kinetic_term, sc_terms[0], sc_terms[1]])
            # compare
            self.assertAlmostEqual(spla.norm(H.as_matrix() - Href), 0)



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


if __name__ == '__main__':
    unittest.main()
