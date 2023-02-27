import numpy as np
import unittest
import qib


class TestSimulator(unittest.TestCase):

    def test_basic_simulation(self):
        """
        Test statevector and tensor network simulators.
        """
        field1 = qib.field.Field(qib.field.ParticleType.QUBIT, qib.lattice.IntegerLattice((2,)))
        field2 = qib.field.Field(qib.field.ParticleType.QUBIT, qib.lattice.IntegerLattice((3,)))
        qa = qib.field.Qubit(field1, 1)
        qb = qib.field.Qubit(field2, 1)
        # Hadamard gate
        hadamard = qib.HadamardGate(qa)
        # CNOT gate
        cnot = qib.ControlledGate(qib.PauliXGate(qb), 1).set_control(qa)
        # construct a simple quantum circuit
        circuit = qib.Circuit()
        circuit.append_gate(hadamard)
        circuit.append_gate(cnot)
        self.assertTrue(circuit.fields() == [field1, field2])
        # reference output: Bell state
        psi_ref = np.transpose(
            np.kron([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)],
                    [1, 0, 0, 0, 0, 0, 0, 0]).reshape(5 * (2,)),
                               (2, 0, 3, 1, 4)).reshape(2**5)
        # statevector simulator
        psi_out = qib.simulator.StatevectorSimulator().run(circuit, [field1, field2], None)
        self.assertTrue(np.allclose(psi_out, psi_ref))
        # tensor network simulator
        tens_out = qib.simulator.TensorNetworkSimulator().run(circuit, [field1, field2], None)
        self.assertTrue(np.allclose(tens_out.reshape(-1), psi_ref))


if __name__ == "__main__":
    unittest.main()
