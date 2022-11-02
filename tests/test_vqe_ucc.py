import numpy as np
import unittest
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
        opt = qib.util.Optimizer(x0=None, method="COBYLA", tol=1e-6)
        # it doesn't work that well with double excitations...
        ans = qib.ansatz.qUCC(field, excitations="s", embedding="jordan_wigner")
        solv = qib.algorithm.VQE(ansatz=ans, optimizer=opt, initial_state=[0]*5 + [1] + [0]*(2**n-1-5), measure_method="statevector")
        #self.assertTrue(solv.run(pauli_ham).success)
        print(solv.run(pauli_ham))


if __name__ == "__main__":
    unittest.main()
