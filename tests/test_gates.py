import unittest
from copy import copy
import numpy as np
from scipy.linalg import expm, block_diag
from scipy import sparse
from scipy.stats import unitary_group
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
        self.assertTrue(np.array_equal(
            X.as_matrix() @ Y.as_matrix(), 1j*Z.as_matrix()))
        self.assertTrue(np.allclose(H.as_matrix() @ X.as_matrix()
                        @ H.as_matrix(), Z.as_matrix()))
        # create a qubit the gates can act on
        field = qib.field.Field(qib.field.ParticleType.QUBIT,
                                qib.lattice.IntegerLattice((5,), pbc=False))
        q = qib.field.Qubit(field, 3)
        for gate in [X, Y, Z, H]:
            self.assertTrue(gate.is_unitary())
            self.assertTrue(gate.is_hermitian())
            self.assertEqual(gate.num_wires, 1)
            gate.on(q)
            self.assertTrue(gate.fields() == [field])
            self.assertTrue(np.array_equal(gate.as_circuit_matrix([field]).toarray(),
                                           np.kron(np.kron(np.identity(8), gate.as_matrix()), np.identity(2))))
            self.assertTrue(np.array_equal(
                gate.as_tensornet().contract_einsum()[0], gate.as_matrix()))
            self.assertIn(gate.as_openQASM()[
                          'name'], [qib.util.GATE_X, qib.util.GATE_Y, qib.util.GATE_Z, qib.util.GATE_H])
            self.assertEqual(gate.as_openQASM()['qubits'], [3])
            g_copy = copy(gate)
            self.assertTrue(g_copy == gate)
            self.assertTrue(np.allclose(g_copy.as_matrix(), gate.as_matrix()))

    def test_rotation_gates(self):
        """
        Test implementation of rotation gates.
        """
        rng = np.random.default_rng()
        # create a qubit the gates can act on
        field = qib.field.Field(qib.field.ParticleType.QUBIT,
                                qib.lattice.IntegerLattice((5,), pbc=False))
        q = qib.field.Qubit(field, 3)
        θ = rng.uniform(0, 2*np.pi)
        nθ = rng.normal(size=3)
        gates = [qib.RxGate(θ, q),
                 qib.RyGate(θ, q),
                 qib.RzGate(θ, q),
                 qib.RotationGate(nθ, q)]
        gates2ang = [qib.RxGate(2*θ, q),
                     qib.RyGate(2*θ, q),
                     qib.RzGate(2*θ, q),
                     qib.RotationGate(2*nθ, q)]
        for i in range(4):
            gate = gates[i]
            self.assertTrue(gate.is_unitary())
            self.assertFalse(gate.is_hermitian())
            self.assertEqual(gate.num_wires, 1)
            self.assertTrue(gate.fields() == [field])
            self.assertTrue(np.allclose(gate.as_matrix() @ gate.inverse().as_matrix(),
                                        np.identity(2)))
            self.assertTrue(np.array_equal(gate.as_circuit_matrix([field]).toarray(),
                                           np.kron(np.kron(np.identity(8), gate.as_matrix()), np.identity(2))))
            self.assertTrue(np.allclose(gate.as_matrix() @ gate.as_matrix(),
                                        gates2ang[i].as_matrix()))
            self.assertTrue(np.array_equal(
                gate.as_tensornet().contract_einsum()[0], gate.as_matrix()))
            self.assertIn(gate.as_openQASM()[
                'name'], [qib.util.GATE_RX, qib.util.GATE_RY, qib.util.GATE_RZ, qib.util.GATE_R])
            self.assertEqual(gate.as_openQASM()['params'], [
                θ] if i < 3 else list(nθ))
            self.assertEqual(gate.as_openQASM()['qubits'], [3])
            if i < 3:
                vθ = np.zeros(3)
                vθ[i] = θ
                self.assertTrue(np.allclose(gate.as_matrix(),
                                            qib.RotationGate(vθ).as_matrix()))
            g_copy = copy(gate)
            self.assertTrue(g_copy == gate)
            self.assertTrue(np.allclose(g_copy.as_matrix(), gate.as_matrix()))
        H = qib.HadamardGate()
        H_mat = H.as_matrix()
        self.assertTrue(np.allclose(H_mat @ gates[0].as_matrix() @ H_mat,
                                    gates[2].as_matrix()))
        g_copy = copy(H)
        self.assertTrue(g_copy == H)
        self.assertTrue(np.allclose(g_copy.as_matrix(), H_mat))

    def test_phase_gates(self):
        """
        Test implementation of S and T gates.
        """
        # create a qubit the gates can act on
        field = qib.field.Field(qib.field.ParticleType.QUBIT,
                                qib.lattice.IntegerLattice((5,), pbc=False))
        q = qib.field.Qubit(field, 3)
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
            self.assertTrue(np.array_equal(gate.as_circuit_matrix([field]).toarray(),
                                           np.kron(np.kron(np.identity(8), gate.as_matrix()), np.identity(2))))
            self.assertTrue(np.array_equal(
                gate.as_tensornet().contract_einsum()[0], gate.as_matrix()))
            self.assertIn(gate.as_openQASM()[
                'name'], [qib.util.GATE_S, qib.util.GATE_T, qib.util.GATE_SDG, qib.util.GATE_TDG])
            self.assertEqual(gate.as_openQASM()['qubits'], [3])
            g_copy = copy(gate)
            self.assertTrue(g_copy == gate)
            self.assertTrue(np.allclose(g_copy.as_matrix(), gate.as_matrix()))
        self.assertTrue(np.allclose(S.as_matrix() @ S.as_matrix(),
                                    qib.PauliZGate().as_matrix()))
        self.assertTrue(np.allclose(T.as_matrix() @ T.as_matrix(),
                                    S.as_matrix()))

    def test_phase_factor_gate(self):
        """
        Test implementation of the phase factor gate.
        """
        rng = np.random.default_rng()
        gate = qib.PhaseFactorGate(rng.normal(), 3)
        self.assertTrue(gate.is_unitary())
        self.assertFalse(gate.is_hermitian())
        self.assertEqual(gate.num_wires, 3)
        gmat = gate.as_matrix()
        self.assertTrue(np.allclose(
            gmat, np.exp(1j*gate.phi) * np.identity(8)))
        self.assertTrue(np.allclose(gmat @ gate.inverse().as_matrix(),
                                    np.identity(8)))
        field = qib.field.Field(qib.field.ParticleType.QUBIT,
                                qib.lattice.IntegerLattice((5,), pbc=False))
        qa = qib.field.Qubit(field, 0)
        qb = qib.field.Qubit(field, 3)
        qc = qib.field.Qubit(field, 2)
        gate.on((qa, qb, qc))
        self.assertTrue(gate.fields() == [field])
        self.assertTrue(np.array_equal(gate.as_circuit_matrix([field]).toarray(),
                                       qib.util.permute_gate_wires(np.kron(np.identity(4), gmat), [0, 3, 2, 1, 4])))
        self.assertTrue(np.allclose(np.reshape(
            gate.as_tensornet().contract_einsum()[0], (8, 8)), gmat))
        with self.assertRaises(NotImplementedError):
            gate.as_openQASM()
        g_copy = copy(gate)
        self.assertTrue(g_copy == gate)
        self.assertTrue(np.allclose(g_copy.as_matrix(), gate.as_matrix()))

    def test_prepare_gate(self):
        """
        Test implementation of the "prepare" gate.
        """
        rng = np.random.default_rng()
        for tp in [False, True]:
            gate = qib.PrepareGate(rng.normal(size=8), 3, transpose=tp)
            # must be normalized
            self.assertTrue(np.allclose(np.linalg.norm(gate.vec, ord=1), 1))
            self.assertTrue(gate.is_unitary())
            self.assertFalse(gate.is_hermitian())
            self.assertEqual(gate.num_wires, 3)
            self.assertTrue(np.allclose(gate.as_matrix() @ gate.inverse().as_matrix(),
                                        np.identity(8)))
            gmat = gate.as_matrix()
            self.assertTrue(np.allclose(gmat[:, 0] if not tp else gmat[0, :], np.sign(
                gate.vec) * np.sqrt(np.abs(gate.vec))))
            field = qib.field.Field(qib.field.ParticleType.QUBIT,
                                    qib.lattice.IntegerLattice((5,), pbc=False))
            qa = qib.field.Qubit(field, 0)
            qb = qib.field.Qubit(field, 3)
            qc = qib.field.Qubit(field, 2)
            gate.on((qa, qb, qc))
            self.assertTrue(gate.fields() == [field])
            self.assertTrue(np.array_equal(gate.as_circuit_matrix([field]).toarray(),
                                           qib.util.permute_gate_wires(np.kron(gmat, np.identity(4)), [0, 3, 2, 1, 4])))
            e1 = np.zeros(8)
            e1[0] = 1
            x = np.sign(gate.vec) * np.sqrt(np.abs(gate.vec))
            self.assertTrue(np.allclose(np.reshape(gate.as_tensornet().contract_einsum()[0], (8, 8)),
                                        np.outer(x, e1) if not tp else np.outer(e1, x)))
            with self.assertRaises(NotImplementedError):
                gate.as_openQASM()
            g_copy = copy(gate)
            self.assertTrue(g_copy == gate)
            self.assertTrue(np.allclose(g_copy.as_matrix(), gate.as_matrix()))

    def test_controlled_gate(self):
        """
        Test implementation of controlled quantum gates.
        """
        rng = np.random.default_rng()

        for cs in [0, 1]:
            # construct the CNOT gate
            cnot = qib.ControlledGate(qib.PauliXGate(), 1, ctrl_state=[cs])
            self.assertEqual(cnot.num_wires, 2)
            self.assertEqual(cnot.num_controls, 1)
            self.assertTrue(np.array_equal(cnot.as_matrix(), np.identity(
                4)[[0, 1, 3, 2]] if cs == 1 else np.identity(4)[[1, 0, 2, 3]]))
            field1 = qib.field.Field(qib.field.ParticleType.QUBIT,
                                     qib.lattice.IntegerLattice((3,), pbc=False))
            qa = qib.field.Qubit(field1, 0)
            qb = qib.field.Qubit(field1, 2)
            cnot.set_control(qb)
            cnot.target_gate().on(qa)
            self.assertTrue(cnot.fields() == [field1])
            self.assertTrue(np.array_equal(cnot.as_circuit_matrix([field1]).toarray(),
                                           qib.util.permute_gate_wires(np.kron(cnot.as_matrix(), np.identity(2)), [1, 2, 0])))
            # additional field
            field2 = qib.field.Field(qib.field.ParticleType.QUBIT,
                                     qib.lattice.IntegerLattice((2,), pbc=False))
            qc = qib.field.Qubit(field2, 1)
            cnot.target_gate().on(qc)
            cnot.set_control(qa)
            self.assertEqual(cnot.num_wires, 2)
            self.assertTrue(cnot.fields() == [field1, field2])
            self.assertTrue(np.array_equal(cnot.as_circuit_matrix([field2, field1]).toarray(),
                                           qib.util.permute_gate_wires(np.kron(cnot.as_matrix(), np.identity(8)), [2, 1, 0, 3, 4])))
            cnot_tensor, axes_map = cnot.as_tensornet().contract_einsum()
            self.assertEqual(cnot.as_openQASM()['name'], qib.util.GATE_CX)
            self.assertEqual(cnot.as_openQASM()['qubits'], [0, 1])
            # control axes are identical; form full matrix for comparison
            cnot_tensor = qib.tensor_network.tensor_network.to_full_tensor(
                cnot_tensor, axes_map)
            self.assertTrue(np.allclose(np.reshape(
                cnot_tensor, (4, 4)), cnot.as_matrix()))
            g_copy = copy(cnot)
            self.assertTrue(g_copy == cnot)
            self.assertTrue(np.allclose(g_copy.as_matrix(), cnot.as_matrix()))

        # construct a Toffoli-like gate, activated by |10>
        toffoli = qib.ControlledGate(qib.PauliXGate(), 2, ctrl_state=[1, 0])
        self.assertEqual(toffoli.num_wires, 3)
        self.assertEqual(toffoli.num_controls, 2)
        self.assertTrue(np.array_equal(toffoli.as_matrix(),
                        np.identity(8)[[0, 1, 2, 3, 5, 4, 6, 7]]))
        self.assertTrue(np.allclose(toffoli.as_matrix() @
                        toffoli.inverse().as_matrix(), np.identity(8)))
        toffoli.set_control(qc, qb)
        toffoli.target_gate().on(qa)
        self.assertTrue(toffoli.fields() == [field2, field1])
        self.assertTrue(np.array_equal(toffoli.as_circuit_matrix([field2, field1]).toarray(),
                                       qib.util.permute_gate_wires(np.kron(toffoli.as_matrix(), np.identity(4)), [3, 0, 2, 4, 1])))
        toffoli_tensor, axes_map = toffoli.as_tensornet().contract_einsum()
        self.assertEqual(toffoli.as_openQASM()['name'], qib.util.GATE_TOFFOLI)
        self.assertEqual(toffoli.as_openQASM()['qubits'], [1, 2, 0])
        # some control axes are identical; form full matrix for comparison
        toffoli_tensor = qib.tensor_network.tensor_network.to_full_tensor(
            toffoli_tensor, axes_map)
        self.assertTrue(np.array_equal(np.reshape(
            toffoli_tensor, (8, 8)), toffoli.as_matrix()))
        g_copy = copy(toffoli)
        self.assertTrue(g_copy == toffoli)
        self.assertTrue(np.allclose(g_copy.as_matrix(), toffoli.as_matrix()))

        # controlled Toffoli gate
        for cs in [0, 1]:
            ctoffoli = qib.ControlledGate(toffoli, 1, ctrl_state=[cs])
            self.assertEqual(ctoffoli.num_wires, 4)
            self.assertEqual(ctoffoli.num_controls, 1)
            ctoffoli_tensor, axes_map = ctoffoli.as_tensornet().contract_einsum()
            with self.assertRaises(NotImplementedError):
                ctoffoli.as_openQASM()
            # some control axes are identical; form full matrix for comparison
            ctoffoli_tensor = qib.tensor_network.tensor_network.to_full_tensor(
                ctoffoli_tensor, axes_map)
            self.assertTrue(np.array_equal(np.reshape(
                ctoffoli_tensor, (16, 16)), ctoffoli.as_matrix()))

        # controlled time evolution gate
        # construct a simple Hamiltonian
        latt = qib.lattice.IntegerLattice((5,), pbc=False)
        field3 = qib.field.Field(qib.field.ParticleType.FERMION, latt)
        # field operator term
        coeffs = qib.util.crandn((5, 5), rng)
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
        cexph_mat_ref = (np.kron(np.diag([1., 0.]), np.identity(2**5))
                         + np.kron(np.diag([0., 1.]), expm(-1j*t*h.as_matrix().toarray())))
        self.assertTrue(np.allclose(cexph.as_matrix(), cexph_mat_ref))
        cexph.set_control(qc)
        self.assertTrue(cexph.fields() == [field2, field3])
        self.assertTrue(np.array_equal(cexph.as_circuit_matrix([field3, field2]).toarray(),
                                       qib.util.permute_gate_wires(np.kron(cexph_mat_ref, np.identity(2)), [1, 2, 3, 4, 5, 6, 0])))
        # inverse
        cexph_inverse = qib.ControlledGate(qib.TimeEvolutionGate(h, -t), 1)
        cexph_inverse.set_control(qc)
        self.assertTrue(np.array_equal(cexph.inverse().as_circuit_matrix([field2, field3]).toarray(),
                                       cexph_inverse.as_circuit_matrix([field2, field3]).toarray()))
        cexph_tensor, axes_map = cexph.as_tensornet().contract_einsum()
        with self.assertRaises(NotImplementedError):
            cexph.as_openQASM()
        # some control axes are identical; form full matrix for comparison
        cexph_tensor = qib.tensor_network.tensor_network.to_full_tensor(
            cexph_tensor, axes_map)
        self.assertTrue(np.allclose(np.reshape(
            cexph_tensor, (2**6, 2**6)), cexph.as_matrix()))
        g_copy = copy(cexph)
        self.assertTrue(g_copy == cexph)
        self.assertTrue(np.allclose(g_copy.as_matrix(), cexph.as_matrix()))

    def test_two_qubit_rotation_gates(self):
        """
        Test implementation of two qubit rotation gates.
        """
        # prepare lattice and qubits
        field = qib.field.Field(qib.field.ParticleType.QUBIT,
                                qib.lattice.IntegerLattice((5,), pbc=False))
        q1 = qib.field.Qubit(field, 2)
        q2 = qib.field.Qubit(field, 3)
        # sample random theta value
        rng = np.random.default_rng()
        θ = rng.uniform(0, 2*np.pi)
        # Rxx gate tests
        X = qib.PauliXGate().as_matrix()
        self.assertTrue(np.allclose(qib.RxxGate(0, q1, q2).as_matrix(), np.identity(4)))
        self.assertTrue(np.allclose(qib.RxxGate(np.pi, q1, q2).as_matrix(), -1j*np.kron(X,X)))
        self.assertTrue(np.allclose(qib.RxxGate(θ, q1, q2).as_matrix(), expm(-1j*θ/2*np.kron(X,X))))
        # Ryy gate tests
        Y = qib.PauliYGate().as_matrix()
        self.assertTrue(np.allclose(qib.RyyGate(0, q1, q2).as_matrix(), np.identity(4)))
        self.assertTrue(np.allclose(qib.RyyGate(np.pi, q1, q2).as_matrix(), -1j*np.kron(Y,Y)))
        self.assertTrue(np.allclose(qib.RyyGate(θ, q1, q2).as_matrix(), expm(-1j*θ/2*np.kron(Y,Y))))
        # Rzz gate tests
        Z = qib.PauliZGate().as_matrix()
        self.assertTrue(np.allclose(qib.RzzGate(0, q1, q2).as_matrix(), np.identity(4)))
        self.assertTrue(np.allclose(qib.RzzGate(2*np.pi, q1, q2).as_matrix(), -np.identity(4)))
        self.assertTrue(np.allclose(qib.RzzGate(np.pi, q1, q2).as_matrix(), -1j*np.kron(Z,Z)))
        RZ = qib.RzGate(θ).as_matrix()
        nRZ = qib.RzGate(-θ).as_matrix()
        self.assertTrue(np.allclose(qib.RzzGate(θ, q1, q2).as_matrix(), block_diag(RZ, nRZ)))
        self.assertTrue(np.allclose(qib.RzzGate(θ, q1, q2).as_matrix(), expm(-1j*θ/2*np.kron(Z,Z))))
        # general tests
        gates = [qib.RxxGate(θ, q1, q2),
                  qib.RzzGate(θ, q1, q2)]
        for i in range(len(gates)):
            gate = gates[i]
            self.assertTrue(gate.is_unitary())
            self.assertFalse(gate.is_hermitian())
            self.assertEqual(gate.num_wires, 2)
            self.assertTrue(gate.fields() == [field])
            self.assertTrue(np.allclose(gate.as_matrix() @ gate.inverse().as_matrix(),
                            np.identity(4)))
            self.assertTrue(np.array_equal(gate.as_tensornet().contract_einsum()[0],
                            gate.as_matrix()))
            self.assertTrue(np.array_equal(gate.as_circuit_matrix([field]).toarray(),
                        np.kron(np.kron(np.eye(2**2), gate.as_matrix()), np.eye(2))))
            g_copy = copy(gate)
            self.assertTrue(g_copy == gate)
            self.assertTrue(np.allclose(g_copy.as_matrix(), gate.as_matrix()))

    def test_multiplexed_gate(self):
        """
        Test implementation of multiplexed quantum gates.
        """
        rng = np.random.default_rng()

        field1 = qib.field.Field(qib.field.ParticleType.QUBIT,
                                 qib.lattice.IntegerLattice((5,), pbc=False))
        qt = qib.field.Qubit(field1, 1)
        tgates = [qib.PauliXGate(qt),
                  qib.RotationGate(rng.normal(size=3), qt),
                  qib.operator.SGate(qt),
                  qib.HadamardGate(qt)]
        # construct a multiplexed gate
        mplxg = qib.MultiplexedGate(tgates, 2)
        self.assertEqual(mplxg.num_wires, 3)
        self.assertEqual(mplxg.num_controls, 2)
        self.assertTrue(np.array_equal(mplxg.as_matrix(),
                        block_diag(*(g.as_matrix() for g in tgates))))
        qa = qib.field.Qubit(field1, 0)
        qb = qib.field.Qubit(field1, 3)
        mplxg.set_control([qa, qb])
        self.assertTrue(mplxg.fields() == [field1])
        self.assertTrue(np.array_equal(mplxg.as_circuit_matrix([field1]).toarray(),
                                       qib.util.permute_gate_wires(np.kron(mplxg.as_matrix(), np.identity(4)), [0, 2, 3, 1, 4])))
        # additional field
        field2 = qib.field.Field(qib.field.ParticleType.QUBIT,
                                 qib.lattice.IntegerLattice((2,), pbc=False))
        qc = qib.field.Qubit(field2, 1)
        mplxg.set_control([qb, qc])
        self.assertEqual(mplxg.num_wires, 3)
        self.assertTrue(mplxg.fields() == [field1, field2])
        self.assertTrue(np.array_equal(mplxg.as_circuit_matrix([field1, field2]).toarray(),
                                       qib.util.permute_gate_wires(np.kron(mplxg.as_matrix(), np.identity(16)), [3, 2, 4, 0, 5, 6, 1])))
        mplxg_tensor, axes_map = mplxg.as_tensornet().contract_einsum()
        with self.assertRaises(NotImplementedError):
            mplxg.as_openQASM()
        # some control axes are identical; form full matrix for comparison
        mplxg_tensor = qib.tensor_network.tensor_network.to_full_tensor(
            mplxg_tensor, axes_map)
        self.assertTrue(np.array_equal(np.reshape(
            mplxg_tensor, (8, 8)), mplxg.as_matrix()))
        g_copy = copy(mplxg)
        self.assertTrue(g_copy == mplxg)
        self.assertTrue(np.allclose(g_copy.as_matrix(), mplxg.as_matrix()))

        # multiplexed time evolution gate
        # construct a simple Hamiltonian
        latt = qib.lattice.IntegerLattice((5,), pbc=False)
        field3 = qib.field.Field(qib.field.ParticleType.FERMION, latt)
        h = [None, None]
        for i in range(2):
            # field operator term
            coeffs = qib.util.crandn((5, 5), rng)
            coeffs = 0.5 * (coeffs + coeffs.conj().T)
            term = qib.operator.FieldOperatorTerm(
                [qib.operator.IFODesc(field3, qib.operator.IFOType.FERMI_CREATE),
                 qib.operator.IFODesc(field3, qib.operator.IFOType.FERMI_ANNIHIL)],
                coeffs)
            self.assertTrue(term.is_hermitian())
            h[i] = qib.FieldOperator([term])
        # time
        t = [1.2, 0.7]
        mplxg = qib.MultiplexedGate(
            [qib.TimeEvolutionGate(h[i], t[i]) for i in range(2)], 1)
        self.assertEqual(mplxg.num_wires, 6)
        mplxg_mat_ref = sum(np.kron(np.diag(np.identity(2)[:, i]),
                                    expm(-1j*t[i]*h[i].as_matrix().toarray())) for i in range(2))
        self.assertTrue(np.allclose(mplxg.as_matrix(), mplxg_mat_ref))
        mplxg.set_control(qc)
        self.assertTrue(mplxg.fields() == [field2, field3])
        self.assertTrue(np.array_equal(mplxg.as_circuit_matrix([field2, field3]).toarray(),
                                       qib.util.permute_gate_wires(np.kron(mplxg_mat_ref, np.identity(2)), [6, 0, 1, 2, 3, 4, 5])))
        # inverse
        self.assertTrue(np.allclose(mplxg.inverse().as_matrix()
                        @ mplxg.as_matrix(), np.identity(2**6)))
        mplxg_tensor, axes_map = mplxg.as_tensornet().contract_einsum()
        with self.assertRaises(NotImplementedError):
            mplxg.as_openQASM()
        # some control axes are identical; form full matrix for comparison
        mplxg_tensor = qib.tensor_network.tensor_network.to_full_tensor(
            mplxg_tensor, axes_map)
        self.assertTrue(np.array_equal(np.reshape(
            mplxg_tensor, (2**6, 2**6)), mplxg.as_matrix()))
        g_copy = copy(mplxg)
        self.assertTrue(g_copy == mplxg)
        self.assertTrue(np.allclose(g_copy.as_matrix(), mplxg.as_matrix()))

    def test_time_evolution_gate(self):
        """
        Test the quantum time evolution gate.
        """
        # Hamiltonian parameters
        µ1 = 0.2
        µ2 = -0.5
        J = 0.76
        # construct a simple Hamiltonian
        latt = qib.lattice.IntegerLattice((2,), pbc=False)
        field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
        # field operator term
        term = qib.operator.FieldOperatorTerm(
            [qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_CREATE),
             qib.operator.IFODesc(field, qib.operator.IFOType.FERMI_ANNIHIL)],
            np.array([[µ1, J], [J, µ2]]))
        self.assertTrue(term.is_hermitian())
        h = qib.FieldOperator([term])
        hmat_ref = np.array(
            [[0, 0,  0,  0],
             [0, µ2, J,  0],
             [0, J,  µ1, 0],
             [0, 0,  0,  µ1 + µ2]])
        self.assertTrue(np.allclose(h.as_matrix().toarray(), hmat_ref))
        # time
        t = 1.2
        gate = qib.TimeEvolutionGate(h, t)
        self.assertEqual(gate.num_wires, 2)
        self.assertTrue(gate.fields() == [field])
        self.assertTrue(np.allclose(gate.as_matrix() @ gate.inverse().as_matrix(),
                                    np.identity(2**gate.num_wires)))
        self.assertTrue(np.allclose(gate.as_matrix() @ gate.as_matrix(),
                                    qib.TimeEvolutionGate(h, 2*t).as_matrix()))
        # reference calculation
        r = np.array([J, 0, 0.5*(μ2 - μ1)])
        ω = np.linalg.norm(r)
        r /= ω
        X = qib.PauliXGate().as_matrix()
        Y = qib.PauliYGate().as_matrix()
        Z = qib.PauliZGate().as_matrix()
        inner_block = np.exp(-1j*t*0.5*(μ1 + μ2))*(np.cos(ω*t)*np.identity(2)
                                                   - 1j*np.sin(ω*t)*(r[0]*X + r[1]*Y + r[2]*Z))
        exp_h_ref = block_diag(np.identity(1), inner_block, [
                               [np.exp(-1j*t*(μ1 + μ2))]])
        self.assertTrue(np.allclose(gate.as_matrix(), exp_h_ref))
        self.assertTrue(np.allclose(
            gate.as_circuit_matrix([field]).toarray(), exp_h_ref))
        self.assertTrue(np.allclose(np.reshape(
            gate.as_tensornet().contract_einsum()[0], (4, 4)), gate.as_matrix()))
        with self.assertRaises(NotImplementedError):
            gate.as_openQASM()
        g_copy = copy(gate)
        self.assertTrue(g_copy == gate)
        self.assertTrue(np.allclose(g_copy.as_matrix(), gate.as_matrix()))

    def test_block_encoding_gate(self):
        """
        Test the block encoding gate.
        """
        rng = np.random.default_rng()

        # construct a simple Hamiltonian
        L = 5
        latt = qib.lattice.IntegerLattice((L,), pbc=True)
        field1 = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
        H = qib.operator.HeisenbergHamiltonian(
            field1, rng.normal(size=3), rng.normal(size=3))
        # rescale parameters (effectively rescales overall Hamiltonian)
        scale = 1.25 * np.linalg.norm(H.as_matrix().toarray(), ord=2)
        H.J /= scale
        H.h /= scale
        # auxiliary qubit
        field2 = qib.field.Field(qib.field.ParticleType.QUBIT,
                                 qib.lattice.IntegerLattice((4,), pbc=False))
        q = qib.field.Qubit(field2, 3)
        for method in qib.operator.BlockEncodingMethod:
            gate = qib.BlockEncodingGate(H, method)
            self.assertEqual(gate.num_wires, L + 1)
            self.assertTrue(gate.encoded_operator() is H)
            gmat = gate.as_matrix()
            self.assertTrue(np.allclose(gmat @ gmat.conj().T,
                                        np.identity(2**gate.num_wires)))
            self.assertTrue(np.allclose(gmat @ gate.inverse().as_matrix(),
                                        np.identity(2**gate.num_wires)))
            gate.set_auxiliary_qubits([q])
            self.assertTrue(gate.fields() == [field2, field1])
            # principal quantum state
            ψp = qib.util.crandn(2**L, rng)
            ψp /= np.linalg.norm(ψp)
            # quantum state on auxiliary register
            ψa = np.kron(qib.util.crandn(8, rng), [1, 0])
            ψa /= np.linalg.norm(ψa)
            # overall quantum state
            ψ = np.kron(ψa, ψp)
            # projection |0><0| acting on auxiliary qubit
            Pa = sparse.kron(sparse.kron(sparse.identity(
                8), sparse.diags([1., 0.])), sparse.identity(2**(L)))
            # text block-encoding of Hamiltonian
            self.assertTrue(np.allclose(Pa @ (gate.as_circuit_matrix([field2, field1]) @ ψ),
                                        np.kron(ψa, H.as_matrix() @ ψp)))
            self.assertTrue(np.allclose(gate.as_circuit_matrix(
                [field2, field1]).toarray(), np.kron(np.identity(2**3), gate.as_matrix())))
            with self.assertRaises(NotImplementedError):
                gate.as_openQASM()
            g_copy = copy(gate)
            self.assertTrue(g_copy == gate)
            self.assertTrue(np.allclose(g_copy.as_matrix(), gate.as_matrix()))

    def test_general_gate(self):
        """
        Test implementation of a general (user-defined) quantum gate.
        """
        gate = qib.GeneralGate(unitary_group.rvs(8), 3)
        self.assertTrue(gate.is_unitary())
        self.assertEqual(gate.num_wires, 3)
        self.assertTrue(np.allclose(gate.as_matrix() @ gate.inverse().as_matrix(),
                                    np.identity(8)))
        field = qib.field.Field(qib.field.ParticleType.QUBIT,
                                qib.lattice.IntegerLattice((5,), pbc=False))
        qa = qib.field.Qubit(field, 0)
        qb = qib.field.Qubit(field, 3)
        qc = qib.field.Qubit(field, 2)
        gate.on((qa, qb, qc))
        self.assertTrue(gate.fields() == [field])
        self.assertTrue(np.array_equal(gate.as_circuit_matrix([field]).toarray(),
                                       qib.util.permute_gate_wires(np.kron(gate.as_matrix(), np.identity(4)), [0, 3, 2, 1, 4])))
        self.assertTrue(np.array_equal(np.reshape(
            gate.as_tensornet().contract_einsum()[0], (8, 8)), gate.as_matrix()))
        with self.assertRaises(NotImplementedError):
            gate.as_openQASM()
        g_copy = copy(gate)
        self.assertTrue(g_copy == gate)
        self.assertTrue(np.allclose(g_copy.as_matrix(), gate.as_matrix()))


if __name__ == "__main__":
    unittest.main()
