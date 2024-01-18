import unittest
import qib


class TestMeasurement(unittest.TestCase):
    def test_measurement(self):
        MEASUREMENT = qib.Measurement()
        # create some qubits the operator can act on
        field = qib.field.Field(
            qib.field.ParticleType.QUBIT, qib.lattice.IntegerLattice((3,), pbc=False))
        q1 = qib.field.Qubit(field, 1)
        q2 = qib.field.Qubit(field, 2)
        q3 = qib.field.Qubit(field, 3)
        self.assertFalse(MEASUREMENT.is_unitary())
        self.assertFalse(MEASUREMENT.is_hermitian())
        self.assertEqual(MEASUREMENT.num_wires, 0)
        MEASUREMENT.on([q1, q2, q3], [2, 1, 3])
        self.assertEqual(MEASUREMENT.num_wires, 3)
        self.assertEqual(MEASUREMENT.particles(), [q1, q2, q3])
        self.assertEqual(MEASUREMENT.fields(), [field, field, field])
        self.assertEqual(MEASUREMENT.memory(), [2, 1, 3])
        with self.assertRaises(NotImplementedError):
            MEASUREMENT.as_matrix()
        self.assertEqual(MEASUREMENT.as_openQASM()['name'], 'measure')
        self.assertEqual(MEASUREMENT.as_openQASM()['qubits'], [1, 2, 3])
        self.assertEqual(MEASUREMENT.as_openQASM()['memory'], [2, 1, 3])

    def test_measurement_qubits_only(self):
        MEASUREMENT = qib.Measurement()
        # create some qubits the operator can act on
        field = qib.field.Field(
            qib.field.ParticleType.QUBIT, qib.lattice.IntegerLattice((3,), pbc=False))
        q1 = qib.field.Qubit(field, 1)
        q2 = qib.field.Qubit(field, 2)
        self.assertFalse(MEASUREMENT.is_unitary())
        self.assertFalse(MEASUREMENT.is_hermitian())
        self.assertEqual(MEASUREMENT.num_wires, 0)
        MEASUREMENT.on([q1, q2])
        self.assertEqual(MEASUREMENT.num_wires, 2)
        self.assertEqual(MEASUREMENT.particles(), [q1, q2])
        self.assertEqual(MEASUREMENT.fields(), [field, field])
        self.assertEqual(MEASUREMENT.memory(), [1, 2])
        with self.assertRaises(NotImplementedError):
            MEASUREMENT.as_matrix()
        self.assertEqual(MEASUREMENT.as_openQASM()['name'], 'measure')
        self.assertEqual(MEASUREMENT.as_openQASM()['qubits'], [1, 2])
        self.assertEqual(MEASUREMENT.as_openQASM()['memory'], [1, 2])

    def test_measurement_length_mismatch(self):
        MEASUREMENT = qib.Measurement()
        # create some qubits the operator can act on
        field = qib.field.Field(
            qib.field.ParticleType.QUBIT, qib.lattice.IntegerLattice((3,), pbc=False))
        q1 = qib.field.Qubit(field, 1)
        q2 = qib.field.Qubit(field, 2)
        with self.assertRaises(ValueError):
            MEASUREMENT.on([q1, q2], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
