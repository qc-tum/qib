import numpy as np
from scipy import sparse
import unittest
import qib


class TestJordanWignerEncoding(unittest.TestCase):

    def test_field_operator_encoding(self):
        """
        Test Jordan-Wigner encoding of a fermionic field operator.
        """
        rng = np.random.default_rng()

        # construct a random fermionic field operator with complex coefficients
        latt = qib.lattice.IntegerLattice((2, 3))
        field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
        # terms with differring number of creation and annihilation operators
        terms = [qib.operator.FieldOperatorTerm(
            [qib.operator.IFODesc(field,
                rng.choice((qib.operator.IFOType.FERMI_CREATE,
                            qib.operator.IFOType.FERMI_ANNIHIL))) for n in range(nops)],
            qib.util.crandn(nops * (latt.nsites,), rng)) for nops in range(4)]
        H = qib.FieldOperator(terms)

        # encode Hamiltonian
        P = qib.transform.jordan_wigner_encode_field_operator(H)

        # compare
        self.assertLess(sparse.linalg.norm(H.as_matrix() - P.as_matrix()), 1e-13)


if __name__ == "__main__":
    unittest.main()
