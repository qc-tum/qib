from __future__ import annotations

import abc
from enum import Enum


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
    ) -> None:
        self.id: int = 0
        self.status: ExperimentStatus = ExperimentStatus.INITIALIZING

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
    def json(self) -> dict:
        """
        Get a JSON representation of the experiment (in Qobj syntax).
        """
        pass


class ExperimentResults(abc.ABC):
    pass
