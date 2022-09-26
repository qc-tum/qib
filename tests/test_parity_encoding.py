import numpy as np
from scipy import sparse
import unittest
import qib


class TestParityEncoding(unittest.TestCase):

    def test_field_operator_encoding(self):
        """
        Test Parity encoding of a fermionic field operator.
        """
        # construct a random fermionic field operator with complex coefficients
        latt = qib.lattice.IntegerLattice((2, 3))
        field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
        # terms with differring number of creation and annihilation operators
        terms = [qib.operator.FieldOperatorTerm(
            [qib.operator.IFODesc(field,
                np.random.choice((qib.operator.IFOType.FERMI_CREATE,
                                  qib.operator.IFOType.FERMI_ANNIHIL))) for n in range(nops)],
            qib.util.crandn(nops * (latt.nsites,))) for nops in range(4)]
        H = qib.FieldOperator(terms)

        # encode Hamiltonian
        P = qib.transform.parity_encode_field_operator(H)

        # compare
        self.assertLess(sparse.linalg.norm(H.as_matrix() - P.as_matrix()), 1e-13)


if __name__ == "__main__":
    unittest.main()
