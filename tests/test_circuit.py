import unittest
import numpy as np
import qib


class TestCircuit(unittest.TestCase):

    def test_basic_circuit(self):
        """
        Test basic quantum circuit functionality.
        """
        field1 = qib.field.Field(
            qib.field.ParticleType.QUBIT, qib.lattice.IntegerLattice((2,)))
        field2 = qib.field.Field(
            qib.field.ParticleType.QUBIT, qib.lattice.IntegerLattice((3,)))
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
        circtens, axes_map = circuit.as_tensornet(
            [field1, field2]).contract_einsum()
        # some control axes are identical; form full matrix for comparison
        circtens = qib.tensor_network.tensor_network.to_full_tensor(
            circtens, axes_map)
        self.assertTrue(np.allclose(np.reshape(circtens, (2**5, 2**5)),
                                    circuit.as_matrix([field1, field2]).toarray()))

    def test_circuit_openQASM_serialization(self):
        """
        Test serialization of a basic quantum circuit to a OpenQASM Qobj.
        """
        field = qib.field.Field(
            qib.field.ParticleType.QUBIT, qib.lattice.IntegerLattice((2,)))
        q1 = qib.field.Qubit(field, 1)
        q2 = qib.field.Qubit(field, 2)
        # Hadamard gate
        hadamard = qib.HadamardGate(q1)
        # CNOT gate
        cnot = qib.ControlledGate(qib.PauliXGate(q2), 1).set_control(q1)
        # measure instruction
        measure = qib.MeasureInstruction()
        measure.on([q1, q2])
        # construct a simple quantum circuit
        circuit = qib.Circuit()
        circuit.append_gate(hadamard)
        circuit.append_gate(cnot)
        circuit.append_gate(measure)
        # particles & clbits
        self.assertEqual(circuit.particles(), {q1, q2})
        self.assertEqual(circuit.clbits(), {1, 2})
        # OpenQASM serialization
        self.assertEqual(len(circuit.as_openQASM()), 3)
        self.assertEqual(circuit.as_openQASM()[0]['name'], 'h')
        self.assertEqual(circuit.as_openQASM()[0]['qubits'], [1])
        self.assertEqual(circuit.as_openQASM()[1]['name'], 'cx')
        self.assertEqual(circuit.as_openQASM()[1]['qubits'], [1, 2])
        self.assertEqual(circuit.as_openQASM()[2]['name'], 'measure')
        self.assertEqual(circuit.as_openQASM()[2]['qubits'], [1, 2])
        self.assertEqual(circuit.as_openQASM()[2]['memory'], [1, 2])


if __name__ == "__main__":
    unittest.main()
