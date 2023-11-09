import abc
from typing import Sequence
from qib.circuit import Circuit
from qib.field import Field


class QuantumProvider(abc.ABC):

    @abc.abstractmethod
    def submit(self, circ: Circuit, fields: Sequence[Field], description):
        """
        Submit a quantum circuit to a backend provider,
        returning a "job" object to query the results.
        """
        pass

    @abc.abstractmethod
    def query_results(self, job):
        """
        Query results of a previously submitted job.
        """
        pass
