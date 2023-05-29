import unittest
import numpy as np
import qib


class TestVQE(unittest.TestCase):

    def test_ucc(self):
        """
        Test VQE + qUCC ansatz.
        """
        latt = qib.lattice.IntegerLattice((2, 2))
        n = latt.nsites
        field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
        hamiltonian = qib.operator.FermiHubbardHamiltonian(field, -1., 5., False)
        pauli_ham = qib.transform.jordan_wigner_encode_field_operator(hamiltonian.as_field_operator())

        # BUG: it doesn't work when all entries for x0 are 0.
        opt = qib.algorithms.vqe.Optimizer(x0=None, method="COBYLA", tol=1e-4)
        # it doesn't work that well with double excitations...
        ans = qib.algorithms.vqe.ansatz.qUCC(field, excitations="s", embedding="jordan_wigner")
        state_0 = np.array([1])
        # first n_occ sites occupied
        n_occ = 2
        for _ in range(n_occ):
            state_0 = np.kron(np.array([0, 1]), state_0)
        for _ in range(n-n_occ):
            state_0 = np.kron(np.array([1, 0]), state_0)
        solv = qib.algorithms.vqe.VQE(ansatz=ans, optimizer=opt, initial_state=state_0, measure_method="statevector")
        self.assertTrue(solv.run(pauli_ham).success)
        #print(solv.run(pauli_ham))


if __name__ == "__main__":
    unittest.main()
