import abc
from typing import Sequence
from qib.circuit import Circuit
from qib.field import Field


class Simulator(abc.ABC):

    @abc.abstractmethod
    def run(self, circ: Circuit, fields: Sequence[Field], description):
        """
        Run a quantum circuit simulation.
        """
        pass
