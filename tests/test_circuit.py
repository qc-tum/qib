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
        self.assertTrue(circuit.fields() == [field1, field2] or circuit.fields() == [field2, field1])
        h_cnot = cnot.as_matrix() @ np.kron(hadamard.as_matrix(), np.identity(2))
        self.assertTrue(np.array_equal(circuit.as_matrix([field1, field2]).toarray(),
                                       permute_gate_wires(np.kron(np.identity(8), h_cnot), [4, 0, 1, 3, 2])))


def permute_gate_wires(u: np.ndarray, perm):
    """
    Transpose (permute) the wires of a quantum gate stored as NumPy array.
    """
    nwires = len(perm)
    assert u.shape == (2**nwires, 2**nwires)
    perm = list(perm)
    u = np.reshape(u, (2*nwires) * (2,))
    u = np.transpose(u, perm + [nwires + p for p in perm])
    u = np.reshape(u, (2**nwires, 2**nwires))
    return u


if __name__ == "__main__":
    unittest.main()
