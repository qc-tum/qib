import numpy as np
import unittest
import qib


class TestParityEncoding(unittest.TestCase):

    def test_field_operator_encoding(self):
        """
        Test parity encoding of a fermionic field operator.
        """
        # construct a random fermionic field operator with complex coefficients
        latt = qib.lattice.IntegerLattice((2, 3))
        field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
        # only real and symmetric matrix (TODO: extend)
        coeffs = np.random.standard_normal(2 * (latt.nsites,))
        # symmetrize
        coeffs = 0.5 * (coeffs + coeffs.T)
        term = qib.operator.FieldOperatorTerm(
            [qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_CREATE),
             qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_ANNIHIL)],
            coeffs)
        H = qib.operator.FieldOperator([term])
        self.assertTrue(H.is_hermitian())

        # encode Hamiltonian
        P = qib.transform.parity_encode_field_operator(H)

        H_eig = np.sort(np.linalg.eigvalsh(H.as_matrix().toarray()))
        P_eig = np.sort(np.linalg.eigvalsh(P.as_matrix().toarray()))
        # compare spectra
        self.assertTrue(np.allclose(H_eig, P_eig, rtol=1e-11,atol=1e-14))


if __name__ == "__main__":
    unittest.main()
