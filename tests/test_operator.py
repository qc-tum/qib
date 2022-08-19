import numpy as np
import unittest
import qib


class TestOperator(unittest.TestCase):

    def test_pauli_string(self):
        """
        Test handling of Pauli strings.
        """
        # construct Pauli string
        z = [0, 1, 0, 1, 1]
        x = [1, 1, 0, 1, 0]
        q = 3
        P = qib.operator.PauliString((z, x, q))
        P.as_matrix()
        # reference calculation
        I = np.identity(2)
        X = np.array([[ 0.,  1.], [ 1.,  0.]])
        Y = np.array([[ 0., -1j], [ 1j,  0.]])
        Z = np.array([[ 1.,  0.], [ 0., -1.]])
        Pref = (-1j)**q * np.kron(np.kron(np.kron(np.kron(X, Y), I), Y), Z)
        # compare
        self.assertTrue(np.allclose(P.as_matrix().toarray(), Pref))

    def test_pauli_operator(self):
        """
        Test Pauli operator functionality.
        """
        # construct Pauli strings
        z = [[0, 1, 0, 1, 1], [1, 0, 1, 1, 1], [1, 1, 0, 1, 0]]
        x = [[1, 1, 0, 1, 0], [0, 1, 1, 0, 0], [1, 0, 0, 0, 1]]
        q = [3, 0, 1]
        weights = [-1.52, 0.687, 0.135]
        P = qib.operator.PauliOperator(
            [qib.operator.WeightedPauliString(
                qib.operator.PauliString((z[j], x[j], q[j])),
                weights[j]) for j in range(3)])
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


if __name__ == '__main__':
    unittest.main()
