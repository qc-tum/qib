from __future__ import annotations

import abc
from enum import Enum

from qib.circuit import Circuit
from qib.backend import Options, ProcessorConfiguration


class ExperimentStatus(str, Enum):
    INITIALIZING = 'INITIALIZING'
    QUEUED = 'QUEUED'
    RUNNING = 'RUNNING'
    DONE = 'DONE'
    ERROR = 'ERROR'
    CANCELLED = 'CANCELLED'


class ExperimentType(str, Enum):
    QASM = 'OpenQASM'
    PULSE = 'OpenPulse'


class Experiment(abc.ABC):
    """
    Parent class for a quantum experiment.

    The actual quantum experiment performed on a given quantum processor.
    """

    @abc.abstractmethod
    def __init__(
            self,
            name: str,
            circuit: Circuit,
            options: Options,
            configuration: ProcessorConfiguration,
            type: ExperimentType,
    ) -> None:
        self.name: str = name
        self.circuit = circuit
        self.options: Options = options
        self.type: ExperimentType = type
        self.configuration: ProcessorConfiguration = configuration
        self._initialize()

    @abc.abstractmethod
    def query_status(self) -> ExperimentResults:
        """
        Query the current status of a previously submitted experiment.
        """

    @abc.abstractmethod
    async def wait_for_results(self) -> ExperimentResults:
        """
        Wait for results of a previously submitted experiment.
        """

    @abc.abstractmethod
    def cancel(self) -> ExperimentResults:
        """
        Cancel a previously submitted experiment.
        """

    @abc.abstractmethod
    def as_openQASM(self) -> dict:
        """
        Get the Qobj OpenQASM representation of the experiment.
        """
        
    @abc.abstractmethod
    def _validate(self):
        """
        Validate the experiment in the context of its quantum processor.
        """
        
    def _initialize(self):
        """
        Initialize the experiment.
        """
        self.status: ExperimentStatus = ExperimentStatus.INITIALIZING
        self.instructions: list = self.circuit.as_openQASM()
        self.id: int = 0


class ExperimentResults(abc.ABC):
    """
    Parent class for a quantum experiment results.

    The results of a quantum experiment performed on a given
    quantum processor.
    """
