import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
import unittest
import qib
from qib.transform.compact_encoding import _encode_edge_operator, _encode_vertex_operator


def comm(a, b):
    """
    Commutator [a, b] = a b - b a.
    """
    return a @ b - b @ a


def anti_comm(a, b):
    """
    Anti-commutator {a, b} = a b + b a.
    """
    return a @ b + b @ a


class TestCompactEncoding(unittest.TestCase):

    def test_anti_comm(self):
        """
        Test anti-commutation relations.
        """
        latt = qib.lattice.OddFaceCenteredLattice((2, 3), pbc=False)
        verts = [(0, 0), (0, 1), (1, 1), (1, 0)]
        for i in range(4):
            Ea = _encode_edge_operator(latt, verts[i],           verts[(i + 1) % 4])
            Eb = _encode_edge_operator(latt, verts[(i + 1) % 4], verts[(i + 2) % 4])
            # anti-commutation relations between edge operators
            self.assertEqual(spla.norm(anti_comm(Ea.as_matrix(), Eb.as_matrix())), 0)
            V = _encode_vertex_operator(latt, verts[(i + 1) % 4])
            # anti-commutation relations between an edge and vertex operator
            self.assertEqual(spla.norm(anti_comm(Ea.as_matrix(), V.as_matrix())), 0)
            self.assertEqual(spla.norm(anti_comm(Eb.as_matrix(), V.as_matrix())), 0)

        latt = qib.lattice.OddFaceCenteredLattice((3, 3), pbc=False)
        # anti-commutation relations between an edge and vertex operator
        for ix in range(2):
            for iy in range(2):
                V = _encode_vertex_operator(latt, (ix, iy))
                E = _encode_edge_operator(  latt, (ix, iy), (ix + 1, iy))
                self.assertEqual(spla.norm(anti_comm(E.as_matrix(), V.as_matrix())), 0)
                V = _encode_vertex_operator(latt, (ix, iy))
                E = _encode_edge_operator(  latt, (ix, iy), (ix, iy + 1))
                self.assertEqual(spla.norm(anti_comm(E.as_matrix(), V.as_matrix())), 0)
                V = _encode_vertex_operator(latt,           (ix + 1, iy))
                E = _encode_edge_operator(  latt, (ix, iy), (ix + 1, iy))
                self.assertEqual(spla.norm(anti_comm(E.as_matrix(), V.as_matrix())), 0)
                V = _encode_vertex_operator(latt,           (ix, iy + 1))
                E = _encode_edge_operator(  latt, (ix, iy), (ix, iy + 1))
                self.assertEqual(spla.norm(anti_comm(E.as_matrix(), V.as_matrix())), 0)
        verts = [(1, 1), (1, 2), (2, 2), (2, 1)]
        for i in range(4):
            Ea = _encode_edge_operator(latt, verts[i],           verts[(i + 1) % 4])
            Eb = _encode_edge_operator(latt, verts[(i + 1) % 4], verts[(i + 2) % 4])
            # anti-commutation relations between edge operators
            self.assertEqual(spla.norm(anti_comm(Ea.as_matrix(), Eb.as_matrix())), 0)
            V = _encode_vertex_operator(latt, verts[(i + 1) % 4])
            # anti-commutation relations between an edge and vertex operator
            self.assertEqual(spla.norm(anti_comm(Ea.as_matrix(), V.as_matrix())), 0)
            self.assertEqual(spla.norm(anti_comm(Eb.as_matrix(), V.as_matrix())), 0)

    def test_comm(self):
        """
        Test commutation relations.
        """
        latt = qib.lattice.OddFaceCenteredLattice((3, 3), pbc=False)
        for x in range(2):
            for y in range(2):
                Ea = _encode_edge_operator(latt, (x, y),     (x + 1, y))
                Eb = _encode_edge_operator(latt, (x, y + 1), (x + 1, y + 1))
                Ec = _encode_edge_operator(latt, (x, y),     (x, y + 1))
                Ed = _encode_edge_operator(latt, (x + 1, y), (x + 1, y + 1))
                self.assertEqual(spla.norm(comm(Ea.as_matrix(), Eb.as_matrix())), 0)
                self.assertEqual(spla.norm(comm(Ec.as_matrix(), Ed.as_matrix())), 0)
                Va = _encode_vertex_operator(latt, (x, y))
                Vb = _encode_vertex_operator(latt, (x + 1, y + 1))
                self.assertEqual(spla.norm(comm(Eb.as_matrix(), Va.as_matrix())), 0)
                self.assertEqual(spla.norm(comm(Ed.as_matrix(), Va.as_matrix())), 0)
                self.assertEqual(spla.norm(comm(Ea.as_matrix(), Vb.as_matrix())), 0)
                self.assertEqual(spla.norm(comm(Ec.as_matrix(), Vb.as_matrix())), 0)

    def test_stabilizer(self):
        """
        Test stabilizer relations.
        """
        latt = qib.lattice.OddFaceCenteredLattice((3, 3), pbc=False)
        # "trivial" stabilizer
        E = []
        verts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        for i in range(4):
            E.append(_encode_edge_operator(latt, verts[i], verts[(i + 1) % 4]))
        self.assertEqual(spla.norm((E[0] @ E[1] @ E[2] @ E[3]).as_matrix()
                                   - sparse.identity(2**latt.nsites)), 0)
        # non-trivial stabilizer
        E = []
        verts = [(1, 0), (1, 1), (2, 1), (2, 0)]
        for i in range(4):
            E.append(_encode_edge_operator(latt, verts[i], verts[(i + 1) % 4]))
        # reference operator
        R = qib.operator.PauliString.identity(latt.nsites)
        for v in verts:
            R.set_pauli('Z', latt.coord_to_index(v))
        R.set_pauli('Y', latt.coord_to_index((0.5, 0.5)))
        R.set_pauli('X', latt.coord_to_index((1.5, 1.5)))
        self.assertEqual(spla.norm((E[0] @ E[1] @ E[2] @ E[3]).as_matrix()
                                   - R.as_matrix()), 0)

    def test_field_operator_encoding(self):
        """
        Test encoding of a fermionic field operator.
        """
        rng = np.random.default_rng()

        # construct fermionic field operator
        latt_fermi = qib.lattice.IntegerLattice((3, 3), pbc=False)
        field = qib.field.Field(qib.field.ParticleType.FERMION, latt_fermi)
        coeffs = rng.standard_normal(2 * (latt_fermi.nsites,))
        # symmetrize
        coeffs = 0.5 * (coeffs + coeffs.T)
        # only on-site and nearest neighbors
        coeffs = coeffs * (np.identity(latt_fermi.nsites) + latt_fermi.adjacency_matrix())
        term = qib.operator.FieldOperatorTerm(
            [qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_CREATE),
             qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_ANNIHIL)],
            coeffs)
        Hfield = qib.operator.FieldOperator([term])

        Hfield_mat = Hfield.as_matrix()
        # must be symmetric
        self.assertEqual(spla.norm(Hfield_mat - Hfield_mat.conj().T), 0)
        # eigenvalues
        ϵref = np.linalg.eigvalsh(Hfield_mat.toarray())

        # encode Hamiltonian
        Henc, latt_enc = qib.transform.compact_encode_field_operator(Hfield)
        Henc_mat = Henc.as_matrix()
        # must be symmetric
        self.assertEqual(spla.norm(Henc_mat - Henc_mat.conj().T), 0)

        # stabilizers
        # upper-right even face
        verts = [(0, 1), (1, 1), (1, 2), (0, 2)]
        R1 = qib.operator.PauliString.identity(latt_enc.nsites)
        for v in verts:
            R1.set_pauli('Z', latt_enc.coord_to_index(v))
        R1.set_pauli('X', latt_enc.coord_to_index((0.5, 0.5)))
        R1.set_pauli('Y', latt_enc.coord_to_index((1.5, 1.5)))
        R1 = R1.as_matrix()
        # lower-left even face
        verts = [(1, 0), (1, 1), (2, 1), (2, 0)]
        R2 = qib.operator.PauliString.identity(latt_enc.nsites)
        for v in verts:
            R2.set_pauli('Z', latt_enc.coord_to_index(v))
        R2.set_pauli('Y', latt_enc.coord_to_index((0.5, 0.5)))
        R2.set_pauli('X', latt_enc.coord_to_index((1.5, 1.5)))
        R2 = R2.as_matrix()
        # stabilizers must commute
        self.assertEqual(spla.norm(comm(R1, R2)), 0)
        # define projector onto stabilizer subspaces
        P = 0.25 * (R1 + sparse.identity(2**latt_enc.nsites)) @ (R2 + sparse.identity(2**latt_enc.nsites))
        # consistency checks
        self.assertEqual(spla.norm(P.conj().T - P), 0)
        self.assertEqual(spla.norm(P @ P - P), 0)
        self.assertEqual(spla.norm(R1 @ P - P), 0)
        self.assertEqual(spla.norm(R2 @ P - P), 0)
        # also commutes with Hamiltonian
        self.assertEqual(spla.norm(comm(Henc_mat, P)), 0)
        # eigenstates with eigenvalue 1 of P define stabilized subspace
        p, ψ = np.linalg.eigh(P.toarray())
        ψ = ψ[:, p > 0.5]
        self.assertEqual(ψ.shape[1], 2**latt_fermi.nsites)
        self.assertTrue(np.allclose(ψ.conj().T @ ψ, np.identity(ψ.shape[1]), rtol=1e-11, atol=1e-14))
        self.assertTrue(np.allclose(R1 @ ψ, ψ, rtol=1e-11, atol=1e-14))
        self.assertTrue(np.allclose(R2 @ ψ, ψ, rtol=1e-11, atol=1e-14))

        # project qubit Hamiltonian onto stabilized subspace
        H_proj = ψ.conj().T @ Henc_mat @ ψ
        # must be symmetric
        self.assertTrue(np.allclose(H_proj, H_proj.conj().T, rtol=1e-11, atol=1e-14))

        # compare spectrum
        ϵ = np.linalg.eigvalsh(H_proj)
        self.assertTrue(np.allclose(ϵ, ϵref, rtol=1e-11, atol=1e-14))


if __name__ == "__main__":
    unittest.main()
