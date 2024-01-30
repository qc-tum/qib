import abc
from typing import Sequence

from qib.util import const
from qib.operator import AbstractOperator
from qib.field import Qubit

class ControlInstruction(AbstractOperator):
    """
    Parent class for circuit control instructions.
    
    A control instruction is an operational directive in a quantum program (i.e. circuit),
    that dictates specific actions to be carried out on qubits within a quantum processor.
    """
    
    def is_unitary(self):
        """
        A control instruction is never unitary.
        """
        return False
    
    def is_hermitian(self):
        """
        A control instruction is never Hermitian.
        """
        return False
    
    @property
    @abc.abstractmethod
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this instruction is applied on.
        """
        
    @abc.abstractmethod
    def particles(self):
        """
        Return the list of quantum particles the instruction is applied on.
        """
    
    @abc.abstractmethod
    def fields(self):
        """
        List of all fields appearing in the instruction.
        """
    
    @abc.abstractmethod
    def on(self, qubits: Sequence[Qubit]):
        """
        Apply the instruction on the specified qubits.
        """
    
    def as_matrix(self):
        """
        Instructions in quantum circuits do not have a sparse matrix representation,
        because they are not quantum operations that act on the state of qubits.
        Instead, these instructions serve as control directives within a quantum circuit.
        """
        raise NotImplementedError("Instructions don't have a matrix representation") 

    def as_openQASM(self):
        """
        Generate a Qobj OpenQASM representation of the instruction.
        """
        raise NotImplementedError(
            "Qobj OpenQASM representation not currently supported for this type of instruction")
    
    @abc.abstractmethod
    def __copy__(self):
        """
        Create a copy of the instruction.
        """
    
    @abc.abstractmethod
    def __eq__(self, __value: object) -> bool:
        """
        Check if instructions are equivalent.
        """


class MeasureInstruction(ControlInstruction):
    """
    A measurement instruction for a quantum circuit.

    The measurement instruction in quantum computing converts the quantum information of a qubit
    into classical information by collapsing the qubit's state to one of the basis states,
    typically |0> or |1>, according to the probabilities defined by its quantum state.
    """

    def __init__(self, qubits: Sequence[Qubit] = None, clbits: Sequence[int] = None):
        self._assign_qubits_clbits(qubits, clbits)

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this instruction is applied on.
        """
        if self.qubits:
            return len(self.qubits)
        return 0

    def particles(self):
        """
        Return the list of quantum particles the instruction is applied on.
        """
        if self.qubits:
            return [q for q in self.qubits]
        return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the instruction is applied on.
        """
        if self.qubits:
            return [q.field for q in self.qubits]
        return []

    def memory(self):
        """
        Return the list of memory slots the instruction will store the results in.
        """
        if self.clbits:
            return [c for c in self.clbits]
        return []

    def on(self, qubits: Sequence[Qubit], clbits: Sequence[int] = None):
        """
        Apply the instruction on the specified qubits.
        """
        self._assign_qubits_clbits(qubits, clbits)

        # enable chaining
        return self

    def as_openQASM(self):
        """
        Generate a Qobj OpenQASM representation of the instruction.
        """
        return {
            "name": const.INSTR_MEASURE,
            "qubits": [q.index for q in self.qubits],
            "memory": [c for c in self.clbits]
        }

    def _assign_qubits_clbits(self, qubits: Sequence[Qubit], clbits: Sequence[int]):
        """
        Assign qubits and classical bits
        """
        # check that the number of qubits and classical bits match
        if qubits and clbits:
            if len(qubits) != len(clbits):
                raise ValueError(
                    "Number of qubits and classical bits must match")

        if qubits:
            self.qubits = qubits
            self.clbits = clbits if clbits else [q.index for q in self.qubits]
        else:
            self.qubits = self.clbits = None

    def __copy__(self):
        """
        Create a copy of the instruction.
        """
        return MeasureInstruction(self.qubits, self.clbits)

    def __eq__(self, other):
        """
        Check if instructions are equivalent.
        """
        if type(other) == type(self) and other.qubits == self.qubits and other.clbits == self.clbits:
            return True
        return False


class BarrierInstruction(ControlInstruction):
    """
    A barrier instruction for a quantum circuit.
    
    The barrier instruction prevents the quantum processor from executing
    any further instructions until all qubits have reached the barrier.
    """
    
    def __init__(self, qubits: Sequence[Qubit] = []):
        self.qubits = qubits
    
    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this instruction is applied on.
        If 0, the barrier is applied on all "wires" of the circuit.
        """
        if self.qubits:
            return len(self.qubits)
        return 0
    
    def particles(self):
        """
        Return the list of quantum particles the instruction is applied on.
        If empty, the barrier is applied on all particles of the circuit.
        """
        if self.qubits:
            return [q for q in self.qubits]
        return []
    
    def fields(self):
        """
        List of all fields appearing in the instruction.
        If empty, all fields of the circuit appears in the instruction.
        """
        if self.qubits:
            return [q.field for q in self.qubits]
        return []
    
    def on(self, qubits: Sequence[Qubit] = []):
        """
        Apply the instruction on the specified qubits.
        If left empty, the instruction is applied on all qubits of the circuit.
        """
        self.qubits = qubits
        
        # enable chaining
        return self
    
    def as_openQASM(self):
        """
        Generate a Qobj OpenQASM representation of the instruction.
        """
        return {
            'name': const.INSTR_BARRIER,
            'qubits': [q.index for q in self.qubits]
        }
    
    def __copy__(self):
        """
        Create a copy of the instruction.
        """
        return BarrierInstruction(self.qubits)
    
    def __eq__(self, other: object) -> bool:
        """
        Check if instructions are equivalent.
        """
        if type(other) == type(self) and self.qubits == other.qubits:
            return True
        return False
    
    
class DelayInstruction(ControlInstruction):
    """
    A delay instruction for a quantum circuit.
    
    The delay instruction specifies a waiting period for qubits,
    pausing execution for a defined duration to control the timing of quantum operations.
    """
    
    def __init__(self, duration: float, qubits: Sequence[Qubit] = []):
        self._duration = duration
        self.qubits = qubits
    
    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this instruction is applied on.
        """
        if self.qubits:
            return len(self.qubits)
        return 0
    
    @property
    def duration(self):
        """
        Get the duration of the delay instruction,
        in dt (differential element of time).
        """
        return self._duration
    
    @duration.setter
    def duration(self, value: float):
        """
        Set the duration of the delay instruction,
        in dt (differential element of time).
        """
        self._duration = value
    
    def particles(self):
        """
        Return the list of quantum particles the instruction is applied on.
        """
        if self.qubits:
            return [q for q in self.qubits]
        return []
    
    def fields(self):
        """
        List of all fields appearing in the instruction.
        """
        if self.qubits:
            return [q.field for q in self.qubits]
        return []
    
    def on(self, qubits: Sequence[Qubit] = None):
        """
        Apply the instruction on the specified qubits.
        """
        self.qubits = qubits
        
        # enable chaining
        return self
    
    def as_openQASM(self):
        """
        Generate a Qobj OpenQASM representation of the instruction.
        """
        return {
            'name': const.INSTR_DELAY,
            'qubits': [q.index for q in self.qubits],
            'duration': self.duration # in dt (differential element of time)
        }
    
    def __copy__(self):
        """
        Create a copy of the instruction.
        """
        return DelayInstruction(self.qubits)
    
    def __eq__(self, other: object) -> bool:
        if (type(other) == type(self) and
            self.qubits == other.qubits and
            other.duration == self.duration):
            return True
        return False