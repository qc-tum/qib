import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
import unittest
import qib

 

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


class TestJWEncoding(unittest.TestCase):
    
    def test_field_operator_encoding(self):
        """
        Test encoding of a fermionic field operator.
        """
        # construct fermionic field operator
        latt_fermi = qib.lattice.IntegerLattice((3, 3), pbc=False)
        field = qib.field.Field(qib.field.ParticleType.FERMION, latt_fermi)
        coeffs = np.random.standard_normal(2 * (latt_fermi.nsites,))                  # double coefficient because it's a double sum
        # symmetrize
        coeffs = 0.5 * (coeffs + coeffs.T)
        # only on-site and nearest neighbors
        coeffs = coeffs * (np.identity(latt_fermi.nsites) + latt_fermi.adjacency_matrix())
        term = qib.operator.FieldOperatorTerm(
            [qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_CREATE),
             qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_ANNIHIL)],
            coeffs)
        Hfield = qib.operator.FieldOperator([term])
        self.assertTrue(Hfield.is_hermitian())

        Hfield_mat = Hfield.as_matrix()
        # must be symmetric
        self.assertEqual(spla.norm(Hfield_mat - Hfield_mat.conj().T), 0)
        # eigenvalues
        ϵref = np.linalg.eigvalsh(Hfield_mat.toarray())

        # encode Hamiltonian
        Henc, latt_enc = qib.transform.Jordan_Wigner_encode_field_operator(Hfield)
        Henc_mat = Henc.as_matrix()
        self.assertTrue(Henc.is_hermitian())
        self.assertTrue(isinstance(latt_enc, qib.lattice.IntegerLattice))
        # must be symmetric
        self.assertEqual(spla.norm(Henc_mat - Henc_mat.conj().T), 0)


        # compare spectrum
        ϵ = np.linalg.eigvalsh(Henc_mat.toarray())
        self.assertTrue(np.allclose(ϵ, ϵref, rtol=1e-11, atol=1e-14))


if __name__ == "__main__":
    unittest.main()
