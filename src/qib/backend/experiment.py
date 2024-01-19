from __future__ import annotations

import abc
from enum import Enum
from qib.backend import Options


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

    @abc.abstractmethod
    def __init__(
            self,
            options: Options
    ) -> None:
        self.id: int = 0
        self.status: ExperimentStatus = ExperimentStatus.INITIALIZING
        self.instructions: list = []
        self.options: Options = options

    @abc.abstractmethod
    def query_results(self) -> ExperimentResults:
        """
        Query results of a previously submitted experiment.
        """
        pass

    @abc.abstractmethod
    async def wait_for_results(self) -> ExperimentResults:
        """
        Wait for results of a previously submitted experiment.
        """
        pass

    @abc.abstractmethod
    def as_openQASM(self) -> dict:
        """
        Get the Qobj OpenQASM representation of the experiment.
        """
        pass


class ExperimentResults(abc.ABC):
    pass
