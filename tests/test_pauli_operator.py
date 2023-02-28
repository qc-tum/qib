import numpy as np
from scipy import sparse
import unittest
import qib


class TestPauliOperator(unittest.TestCase):

    def test_pauli_string(self):
        """
        Test handling of Pauli strings.
        """
        rng = np.random.default_rng()

        L = 5
        latt = qib.lattice.IntegerLattice((L,), pbc=True)
        field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
        self.assertEqual(sparse.linalg.norm(qib.PauliString.identity(5).as_matrix()
                                            - sparse.identity(2**5)), 0)
        # construct Pauli string (single paulis and string)
        P = qib.PauliString.from_single_paulis(5, ('Y', 1), ('X', 0), ('Y', 3), ('Z', 4), q=3)
        P.set_field(field)
        self.assertTrue(P == qib.PauliString.from_string("iXYIYZ"))
        self.assertTrue(P.is_unitary())
        self.assertFalse(P.is_hermitian())
        self.assertEqual(str(P), "iXYIYZ")
        self.assertTrue(P.fields() == [field])
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
        P2 = qib.PauliString.from_single_paulis(5, ('Z', 4), ('Y', 0), ('Y', 1), ('X', 2), q=2)
        P2.set_field(field)
        self.assertTrue(P2 == qib.PauliString.from_string("-YYXIZ"))
        self.assertTrue(P2.is_unitary())
        self.assertTrue(P2.is_hermitian())
        self.assertEqual(str(P2), "-YYXIZ")
        self.assertTrue(P2.fields() == [field])
        # logical product
        self.assertEqual(sparse.linalg.norm(( P @ P2).as_matrix()
                                            - P.as_matrix() @ P2.as_matrix()), 0)
        # logical product for various lengths
        for nqubits in range(1, 10):
            Plist = []
            for j in range(2):
                z = rng.integers(0, 2, nqubits)
                x = rng.integers(0, 2, nqubits)
                q = rng.integers(0, 4)
                Plist.append(qib.PauliString(z, x, q))
            self.assertEqual(sparse.linalg.norm( (Plist[0] @ Plist[1]).as_matrix()
                                                - Plist[0].as_matrix() @ Plist[1].as_matrix()), 0)

    def test_pauli_operator(self):
        """
        Test Pauli operator functionality.
        """
        L = 5
        latt = qib.lattice.IntegerLattice((L,), pbc=True)
        field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
        # construct Pauli strings
        z = [[0, 1, 0, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 0]]
        x = [[1, 1, 0, 1, 0], [0, 1, 1, 0, 0], [1, 0, 0, 0, 1]]
        q = [3, 0, 1]
        weights = [-1.52, 0.687, 0.135]
        P = qib.PauliOperator(
            [qib.WeightedPauliString(
                qib.PauliString(z[j], x[j], q[j]),
                weights[j]) for j in range(3)])
        P.set_field(field)
        self.assertEqual(P.num_qubits, 5)
        self.assertFalse(P.is_hermitian())
        self.assertTrue(P.fields() == [field])
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

        # check summation of Pauli operators: first and last Pauli string
        # are the same as above, only with different weights
        z_add = [[0, 1, 0, 1, 1], [0, 0, 1, 1, 1], [1, 1, 0, 1, 0]]
        x_add = [[1, 1, 0, 1, 0], [0, 1, 1, 0, 0], [1, 0, 0, 0, 1]]
        q_add = [3, 0, 1]
        weights_add = [0.52, 1.87, -0.135]
        wps_list = [
            qib.WeightedPauliString(
                qib.PauliString(z_add[j], x_add[j], q_add[j]),
                weights_add[j]) for j in range(3)]
        P.add_pauli_string(wps_list[0])
        self.assertEqual(len(P.pstrings), 3)
        self.assertTrue(P.pstrings[0].weight == weights[0] + weights_add[0])
        P.add_pauli_string(wps_list[1])
        self.assertEqual(len(P.pstrings), 4)
        P.add_pauli_string(wps_list[2])
        self.assertEqual(len(P.pstrings), 4)
        self.assertTrue(P.pstrings[2].weight == weights[2] + weights_add[2])


if __name__ == "__main__":
    unittest.main()
