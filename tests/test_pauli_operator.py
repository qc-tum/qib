import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
import unittest
import qib


class TestPauliOperator(unittest.TestCase):

    def test_pauli_string(self):
        """
        Test handling of Pauli strings.
        """
        self.assertEqual(spla.norm(qib.PauliString.identity(5).as_matrix()
                                   - sparse.identity(2**5)), 0)
        # construct Pauli string (single paulis and string)
        P = qib.PauliString.from_single_paulis(5, ('Y', 1), ('X', 0), ('Y', 3), ('Z', 4), q=3)
        self.assertTrue(P == qib.PauliString.from_string("iXYIYZ"))
        self.assertTrue(P.is_unitary())
        self.assertFalse(P.is_hermitian())
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
        self.assertTrue(P2 == qib.PauliString.from_string("-YYXIZ"))
        self.assertTrue(P2.is_unitary())
        self.assertTrue(P2.is_hermitian())
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
                Plist.append(qib.PauliString(z, x, q))
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
        P = qib.PauliOperator(
            [qib.operator.WeightedPauliString(
                qib.PauliString(z[j], x[j], q[j]),
                weights[j]) for j in range(3)])
        self.assertEqual(P.num_qubits, 5)
        self.assertFalse(P.is_hermitian())
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


if __name__ == "__main__":
    unittest.main()
