import numpy as np
from scipy.linalg import block_diag
import unittest
import qib
import qib.operator as qop


class TestGates(unittest.TestCase):

    def test_basic_gates(self):
        """
        Test implementation of basic quantum gates.
        """
        X = qop.PauliXGate()
        Y = qop.PauliYGate()
        Z = qop.PauliZGate()
        H = qop.HadamardGate()
        for g in [X, Y, Z, H]:
            self.assertTrue(g.is_unitary())
            self.assertTrue(g.is_hermitian())
            self.assertEqual(g.num_wires, 1)
        self.assertTrue(np.array_equal(X.as_matrix() @ Y.as_matrix(), 1j*Z.as_matrix()))
        self.assertTrue(np.allclose(H.as_matrix() @ X.as_matrix() @ H.as_matrix(), Z.as_matrix()))

    def test_time_evolution_gate(self):
        """
        Test the quantum time evolution gate.
        """
        # Hamiltonian parameters
        μ1 =  0.2
        μ2 = -0.5
        J = 0.76
        # construct a simple Hamiltonian
        latt = qib.lattice.IntegerLattice((2,), pbc=False)
        field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
        # field operator term
        term = qop.FieldOperatorTerm(
            [qop.IFODesc(field, qop.IFOType.FERMI_CREATE),
             qop.IFODesc(field, qop.IFOType.FERMI_ANNIHIL)],
            np.array([[μ1, J], [J, μ2]]))
        self.assertTrue(term.is_hermitian())
        h = qop.FieldOperator([term])
        hmat_ref = np.array(
            [[0, 0,  0,  0      ],
             [0, μ1, J,  0      ],
             [0, J,  μ2, 0      ],
             [0, 0,  0,  μ1 + μ2]])
        self.assertTrue(np.allclose(h.as_matrix().toarray(), hmat_ref))
        # time
        t = 1.2
        gate = qop.TimeEvolutionGate(h, t)
        self.assertEqual(gate.num_wires, 2)
        # reference calculation
        r = np.array([J, 0, 0.5*(μ1 - μ2)])
        ω = np.linalg.norm(r)
        r /= ω
        X = qop.PauliXGate().as_matrix()
        Y = qop.PauliYGate().as_matrix()
        Z = qop.PauliZGate().as_matrix()
        inner_block = np.exp(-1j*t*0.5*(μ1 + μ2))*(np.cos(ω*t)*np.identity(2)
                                              - 1j*np.sin(ω*t)*(r[0]*X + r[1]*Y + r[2]*Z))
        exp_h_ref = block_diag(np.identity(1), inner_block, [[np.exp(-1j*t*(μ1 + μ2))]])
        self.assertTrue(np.allclose(gate.as_matrix().toarray(), exp_h_ref))


if __name__ == "__main__":
    unittest.main()
