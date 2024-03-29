import numpy as np
import unittest
import qib


class TestBackend(unittest.TestCase):

    def test_tensor_network(self):
        """
        Test tensor network backend functionality.
        """
        field = qib.field.Field(qib.field.ParticleType.QUBIT, qib.lattice.IntegerLattice((2,)))
        qa = qib.field.Qubit(field, 0)
        qb = qib.field.Qubit(field, 1)
        # Hadamard gate
        hadamard = qib.HadamardGate(qa)
        # CNOT gate
        cnot = qib.ControlledGate(qib.PauliXGate(qb), 1).set_control(qa)
        # construct a simple quantum circuit
        circuit = qib.Circuit()
        circuit.append_gate(hadamard)
        circuit.append_gate(cnot)
        self.assertTrue(circuit.fields() == [field])
        h_cnot = cnot.as_matrix() @ np.kron(hadamard.as_matrix(), np.identity(2))
        self.assertTrue(np.array_equal(circuit.as_matrix([field]).toarray(), h_cnot))
        provider = qib.backend.TensorNetworkProvider()
        provider.submit(circuit, [field], { "filename": "bell_circuit_tensornet.hdf5" })


if __name__ == "__main__":
    unittest.main()
