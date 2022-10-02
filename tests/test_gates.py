import numpy as np
from scipy.linalg import expm, block_diag
import unittest
import qib


class TestGates(unittest.TestCase):

    def test_basic_gates(self):
        """
        Test implementation of basic quantum gates.
        """
        X = qib.PauliXGate()
        Y = qib.PauliYGate()
        Z = qib.PauliZGate()
        H = qib.HadamardGate()
        self.assertTrue(np.array_equal(X.as_matrix() @ Y.as_matrix(), 1j*Z.as_matrix()))
        self.assertTrue(np.allclose(H.as_matrix() @ X.as_matrix() @ H.as_matrix(), Z.as_matrix()))
        # create a qubit the gates can act on
        field = qib.field.Field(qib.field.ParticleType.QUBIT,
                                qib.lattice.IntegerLattice((5,), pbc=False))
        q = qib.field.Qubit(field, 1)
        for gate in [X, Y, Z, H]:
            self.assertTrue(gate.is_unitary())
            self.assertTrue(gate.is_hermitian())
            self.assertEqual(gate.num_wires, 1)
            gate.on(q)
            self.assertTrue(gate.fields() == [field])
            self.assertTrue(np.array_equal(gate._circuit_matrix([field]).toarray(),
                                           np.kron(np.kron(np.identity(8), gate.as_matrix()), np.identity(2))))

    def test_rotation_gates(self):
        """
        Test implementation of rotation gates.
        """
        # create a qubit the gates can act on
        field = qib.field.Field(qib.field.ParticleType.QUBIT,
                                qib.lattice.IntegerLattice((5,), pbc=False))
        q = qib.field.Qubit(field, 1)
        θlist = np.random.uniform(0, 2*np.pi, size=3)
        gates = [qib.RxGate(θlist[0], q),
                 qib.RyGate(θlist[1], q),
                 qib.RzGate(θlist[2], q)]
        gates2ang = [qib.RxGate(2*θlist[0], q),
                     qib.RyGate(2*θlist[1], q),
                     qib.RzGate(2*θlist[2], q)]
        for gate, gate2ang in zip(gates, gates2ang):
            self.assertTrue(gate.is_unitary())
            self.assertFalse(gate.is_hermitian())
            self.assertEqual(gate.num_wires, 1)
            self.assertTrue(gate.fields() == [field])
            self.assertTrue(np.allclose(np.matmul(gate.as_matrix(), gate.inverse().as_matrix()),
                                        np.identity(2)))
            self.assertTrue(np.array_equal(gate._circuit_matrix([field]).toarray(),
                                           np.kron(np.kron(np.identity(8), gate.as_matrix()), np.identity(2))))
            self.assertTrue(np.allclose(gate.as_matrix() @ gate.as_matrix(),
                                        gate2ang.as_matrix()))

    def test_phase_gates(self):
        """
        Test implementation of S and T gates
        """
        # create a qubit the gates can act on
        field = qib.field.Field(qib.field.ParticleType.QUBIT,
                                qib.lattice.IntegerLattice((5,), pbc=False))
        q = qib.field.Qubit(field, 1)
        S = qib.operator.SGate(q)
        T = qib.operator.TGate(q)
        S_adj = qib.operator.SAdjGate(q)
        T_adj = qib.operator.TAdjGate(q)
        for gate in [S, T, S_adj, T_adj]:
            self.assertTrue(gate.is_unitary())
            self.assertFalse(gate.is_hermitian())
            self.assertEqual(gate.num_wires, 1)
            self.assertTrue(gate.fields() == [field])
            self.assertTrue(np.allclose(gate.inverse().as_matrix() @ gate.as_matrix(),
                                        np.identity(2)))
        self.assertTrue(np.allclose(S.as_matrix() @ S.as_matrix(),
                                    qib.PauliZGate().as_matrix()))
        self.assertTrue(np.allclose(T.as_matrix() @ T.as_matrix(),
                                    S.as_matrix()))

    def test_controlled_gate(self):
        """
        Test implementation of controlled quantum gates.
        """
        # construct the CNOT gate
        cnot = qib.ControlledGate(qib.PauliXGate(), 1)
        self.assertEqual(cnot.num_wires, 2)
        self.assertEqual(cnot.num_controls, 1)
        self.assertTrue(np.array_equal(cnot.as_matrix(), np.identity(4)[[0, 1, 3, 2]]))
        field1 = qib.field.Field(qib.field.ParticleType.QUBIT,
                                qib.lattice.IntegerLattice((3,), pbc=False))
        qa = qib.field.Qubit(field1, 0)
        qb = qib.field.Qubit(field1, 2)
        cnot.set_control(qb)
        cnot.target_gate().on(qa)
        self.assertTrue(cnot.fields() == [field1])
        self.assertTrue(np.array_equal(cnot._circuit_matrix([field1]).toarray(),
                                       permute_gate_wires(np.kron(cnot.as_matrix(), np.identity(2)), [0, 2, 1])))
        # additional field
        field2 = qib.field.Field(qib.field.ParticleType.QUBIT,
                                 qib.lattice.IntegerLattice((2,), pbc=False))
        qc = qib.field.Qubit(field2, 1)
        cnot.target_gate().on(qc)
        cnot.set_control(qa)
        self.assertEqual(cnot.num_wires, 2)
        self.assertTrue(cnot.fields() == [field1, field2] or cnot.fields() == [field2, field1])
        self.assertTrue(np.array_equal(cnot._circuit_matrix([field2, field1]).toarray(),
                                       permute_gate_wires(np.kron(np.identity(8), cnot.as_matrix()), [0, 1, 3, 4, 2])))

        # construct the Toffoli gate
        toffoli = qib.ControlledGate(qib.PauliXGate(), 2)
        self.assertEqual(toffoli.num_wires, 3)
        self.assertEqual(toffoli.num_controls, 2)
        self.assertTrue(np.array_equal(toffoli.as_matrix(), np.identity(8)[[0, 1, 2, 3, 4, 5, 7, 6]]))
        toffoli.set_control(qc, qb)
        toffoli.target_gate().on(qa)
        self.assertTrue(toffoli.fields() == [field1, field2] or toffoli.fields() == [field2, field1])
        self.assertTrue(np.array_equal(toffoli._circuit_matrix([field2, field1]).toarray(),
                                       permute_gate_wires(np.kron(np.identity(4), toffoli.as_matrix()), [2, 0, 4, 3, 1])))

        # controlled time evolution gate
        # construct a simple Hamiltonian
        latt = qib.lattice.IntegerLattice((5,), pbc=False)
        field3 = qib.field.Field(qib.field.ParticleType.FERMION, latt)
        # field operator term
        coeffs = qib.util.crandn((5, 5))
        coeffs = 0.5 * (coeffs + coeffs.conj().T)
        term = qib.operator.FieldOperatorTerm(
            [qib.operator.IFODesc(field3, qib.operator.IFOType.FERMI_CREATE),
             qib.operator.IFODesc(field3, qib.operator.IFOType.FERMI_ANNIHIL)],
            coeffs)
        self.assertTrue(term.is_hermitian())
        h = qib.FieldOperator([term])
        # time
        t = 1.2
        cexph = qib.ControlledGate(qib.TimeEvolutionGate(h, t), 1)
        self.assertEqual(cexph.num_wires, 6)
        cexph_mat_ref = (  np.kron(np.diag([1., 0.]), np.identity(2**5))
                         + np.kron(np.diag([0., 1.]), expm(-1j*t*h.as_matrix().toarray())))
        self.assertTrue(np.allclose(cexph.as_matrix(), cexph_mat_ref))
        cexph.set_control(qc)
        self.assertTrue(cexph.fields() == [field2, field3] or cexph.fields() == [field3, field2])
        self.assertTrue(np.array_equal(cexph._circuit_matrix([field2, field3]).toarray(),
                                       permute_gate_wires(np.kron(np.identity(2), cexph_mat_ref), [2, 3, 4, 5, 6, 1, 0])))

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
        term = qib.operator.FieldOperatorTerm(
            [qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_CREATE),
             qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_ANNIHIL)],
            np.array([[μ1, J], [J, μ2]]))
        self.assertTrue(term.is_hermitian())
        h = qib.FieldOperator([term])
        hmat_ref = np.array(
            [[0, 0,  0,  0      ],
             [0, μ1, J,  0      ],
             [0, J,  μ2, 0      ],
             [0, 0,  0,  μ1 + μ2]])
        self.assertTrue(np.allclose(h.as_matrix().toarray(), hmat_ref))
        # time
        t = 1.2
        gate = qib.TimeEvolutionGate(h, t)
        self.assertEqual(gate.num_wires, 2)
        self.assertTrue(gate.fields() == [field])
        self.assertTrue(np.allclose(gate.as_matrix() @ gate.inverse().as_matrix(),
                                    np.identity(2*gate.num_wires)))
        self.assertTrue(np.allclose(gate.as_matrix() @ gate.as_matrix(),
                                    qib.operator.TimeEvolutionGate(h, 2*t).as_matrix()))
        # reference calculation
        r = np.array([J, 0, 0.5*(μ1 - μ2)])
        ω = np.linalg.norm(r)
        r /= ω
        X = qib.PauliXGate().as_matrix()
        Y = qib.PauliYGate().as_matrix()
        Z = qib.PauliZGate().as_matrix()
        inner_block = np.exp(-1j*t*0.5*(μ1 + μ2))*(np.cos(ω*t)*np.identity(2)
                                              - 1j*np.sin(ω*t)*(r[0]*X + r[1]*Y + r[2]*Z))
        exp_h_ref = block_diag(np.identity(1), inner_block, [[np.exp(-1j*t*(μ1 + μ2))]])
        self.assertTrue(np.allclose(gate.as_matrix(), exp_h_ref))
        self.assertTrue(np.allclose(gate._circuit_matrix([field]).toarray(), exp_h_ref))


def permute_gate_wires(u: np.ndarray, perm):
    """
    Transpose (permute) the wires of a quantum gate stored as NumPy array.
    """
    nwires = len(perm)
    assert u.shape == (2**nwires, 2**nwires)
    perm = list(perm)
    u = np.reshape(u, (2*nwires) * (2,))
    u = np.transpose(u, perm + [nwires + p for p in perm])
    u = np.reshape(u, (2**nwires, 2**nwires))
    return u


if __name__ == "__main__":
    unittest.main()
