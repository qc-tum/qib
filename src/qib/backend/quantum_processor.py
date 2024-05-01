import abc

from qib.circuit import Circuit
from qib.backend import ProcessorConfiguration, Experiment, Options


class QuantumProcessor(abc.ABC):
    """
    Parent class for quantum processor (a.k.a. backend).

    A quantum processor is a device that can execute quantum circuits,
    be it a quantum computer, a simulator, or a different type of quantum backend.
    """

    @staticmethod
    @abc.abstractmethod
    def configuration() -> ProcessorConfiguration:
        """
        The configuration of the quantum processor.
        """

    @abc.abstractmethod
    def submit_experiment(self, name: str, circ: Circuit, options: Options) -> Experiment:
        """
        Submit a quantum circuit and experiment execution options to a quantum processor backend,
        returning a validated "experiment" object to query the results.
        """

    @abc.abstractmethod
    def _send_request(self, experiment: Experiment):
        """
        Send already validated experiment to the quantum processor backend.
        """
    
    @abc.abstractmethod
    def _process_response(self, experiment: Experiment, response: dict):
        """
        Process the response received from the backend.
        """