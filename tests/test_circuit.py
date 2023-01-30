import numpy as np
import unittest
import qib


class TestCircuit(unittest.TestCase):

    def test_basic_circuit(self):
        """
        Test basic quantum circuit functionality.
        """
        field1 = qib.field.Field(qib.field.ParticleType.QUBIT, qib.lattice.IntegerLattice((2,)))
        field2 = qib.field.Field(qib.field.ParticleType.QUBIT, qib.lattice.IntegerLattice((3,)))
        qa = qib.field.Qubit(field1, 1)
        qb = qib.field.Qubit(field2, 2)
        # Hadamard gate
        hadamard = qib.HadamardGate(qa)
        # CNOT gate
        cnot = qib.ControlledGate(qib.PauliXGate(qb), 1).set_control(qa)
        # construct a simple quantum circuit
        circuit = qib.Circuit()
        circuit.append_gate(hadamard)
        circuit.append_gate(cnot)
        self.assertTrue(circuit.fields() == [field1, field2])
        h_cnot = cnot.as_matrix() @ np.kron(hadamard.as_matrix(), np.identity(2))
        self.assertTrue(np.array_equal(circuit.as_matrix([field1, field2]).toarray(),
                                       qib.util.permute_gate_wires(np.kron(h_cnot, np.identity(8)), [2, 0, 3, 4, 1])))
        self.assertTrue(np.allclose(circuit.as_matrix([field1, field2]).toarray()
            @ circuit.inverse().as_matrix([field1, field2]).toarray(),
            np.identity(2**5)))
        circtens, axes_map = circuit.as_tensornet([field1, field2]).contract_einsum()
        # some control axes are identical; form full matrix for comparison
        circtens = qib.tensor_network.tensor_network.to_full_tensor(circtens, axes_map)
        self.assertTrue(np.allclose(np.reshape(circtens, (2**5, 2**5)),
                                    circuit.as_matrix([field1, field2]).toarray()))


if __name__ == "__main__":
    unittest.main()
