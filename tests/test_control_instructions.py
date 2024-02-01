import unittest
import qib
from copy import copy


class TestControlInstructions(unittest.TestCase):
    def test_measure(self):
        MEASURE = qib.MeasureInstruction()
        # create some qubits the instruction can be applied on
        field = qib.field.Field(
            qib.field.ParticleType.QUBIT, qib.lattice.IntegerLattice((3,), pbc=False))
        q1 = qib.field.Qubit(field, 1)
        q2 = qib.field.Qubit(field, 2)
        q3 = qib.field.Qubit(field, 3)
        self.assertFalse(MEASURE.is_unitary())
        self.assertFalse(MEASURE.is_hermitian())
        self.assertEqual(MEASURE.num_wires, 0)
        with self.assertRaises(NotImplementedError):
            MEASURE.as_matrix()
        # qubits & clbits
        MEASURE.on([q1, q2, q3], [2, 1, 3])
        self.assertEqual(MEASURE.num_wires, 3)
        self.assertEqual(MEASURE.particles(), [q1, q2, q3])
        self.assertEqual(MEASURE.fields(), [field, field, field])
        self.assertEqual(MEASURE.memory(), [2, 1, 3])
        with self.assertRaises(NotImplementedError):
            MEASURE.as_matrix()
        self.assertEqual(MEASURE.as_openQASM()['name'], 'measure')
        self.assertEqual(MEASURE.as_openQASM()['qubits'], [1, 2, 3])
        self.assertEqual(MEASURE.as_openQASM()['memory'], [2, 1, 3])
        # qubits only
        MEASURE.on([q1, q2])
        self.assertEqual(MEASURE.num_wires, 2)
        self.assertEqual(MEASURE.particles(), [q1, q2])
        self.assertEqual(MEASURE.fields(), [field, field])
        self.assertEqual(MEASURE.memory(), [1, 2])
        self.assertEqual(MEASURE.as_openQASM()['name'], 'measure')
        self.assertEqual(MEASURE.as_openQASM()['qubits'], [1, 2])
        self.assertEqual(MEASURE.as_openQASM()['memory'], [1, 2])
        # length mismatch
        with self.assertRaises(ValueError):
            MEASURE.on([q1, q2], [1, 2, 3])
        # copy & equal
        MEASURE_COPY = copy(MEASURE)
        self.assertEqual(MEASURE_COPY, MEASURE)

    def test_barrier(self):
        BARRIER = qib.BarrierInstruction()
        # create some qubits the instruction can be applied on
        field = qib.field.Field(
            qib.field.ParticleType.QUBIT, qib.lattice.IntegerLattice((3,), pbc=False))
        q1 = qib.field.Qubit(field, 1)
        q2 = qib.field.Qubit(field, 2)
        q3 = qib.field.Qubit(field, 3)
        self.assertFalse(BARRIER.is_unitary())
        self.assertFalse(BARRIER.is_hermitian())
        self.assertEqual(BARRIER.num_wires, 0)
        BARRIER.on([q1, q2, q3])
        self.assertEqual(BARRIER.num_wires, 3)
        self.assertEqual(BARRIER.particles(), [q1, q2, q3])
        self.assertEqual(BARRIER.fields(), [field, field, field])
        with self.assertRaises(NotImplementedError):
            BARRIER.as_matrix()
        self.assertEqual(BARRIER.as_openQASM()['name'], 'barrier')
        self.assertEqual(BARRIER.as_openQASM()['qubits'], [1, 2, 3])
        # copy & equal
        BARRIER_COPY = copy(BARRIER)
        self.assertEqual(BARRIER_COPY, BARRIER)
        # empty qubits list
        BARRIER.on([])
        self.assertEqual(BARRIER.as_openQASM()['qubits'], [])
        
    def test_delay(self):
        DELAY = qib.DelayInstruction(10)
        # create some qubits the instruction can be applied on
        field = qib.field.Field(
            qib.field.ParticleType.QUBIT, qib.lattice.IntegerLattice((3,), pbc=False))
        q1 = qib.field.Qubit(field, 1)
        q2 = qib.field.Qubit(field, 2)
        q3 = qib.field.Qubit(field, 3)
        self.assertFalse(DELAY.is_unitary())
        self.assertFalse(DELAY.is_hermitian())
        self.assertEqual(DELAY.num_wires, 0)
        DELAY.on([q1, q2, q3])
        self.assertEqual(DELAY.num_wires, 3)
        self.assertEqual(DELAY.particles(), [q1, q2, q3])
        self.assertEqual(DELAY.fields(), [field, field, field])
        with self.assertRaises(NotImplementedError):
            DELAY.as_matrix()
        self.assertEqual(DELAY.as_openQASM()['name'], 'delay')
        self.assertEqual(DELAY.as_openQASM()['qubits'], [1, 2, 3])
        self.assertEqual(DELAY.as_openQASM()['duration'], 10)
        # copy & equal
        DELAY_COPY = copy(DELAY)
        self.assertEqual(DELAY_COPY, DELAY)
        # duration
        DELAY_COPY.duration = 20.0
        self.assertEqual(DELAY_COPY.as_openQASM()['duration'], 20)
        self.assertNotEqual(DELAY_COPY, DELAY)
        # empty qubits list
        DELAY.on([])
        self.assertEqual(DELAY.as_openQASM()['qubits'], [])

if __name__ == "__main__":
    unittest.main()
