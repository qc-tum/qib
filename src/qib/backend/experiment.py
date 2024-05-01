from __future__ import annotations

import abc
from enum import Enum

from qib.circuit import Circuit
from qib.backend import Options, ProcessorConfiguration, ProcessorCredentials


class ExperimentStatus(str, Enum):
    INITIALIZING = 'INITIALIZING'
    QUEUED = 'QUEUED'
    RUNNING = 'RUNNING'
    DONE = 'DONE'
    ERROR = 'ERROR'
    CANCELLED = 'CANCELLED'
    
    def is_terminal(self) -> bool:
        """
        Check if the experiment status is terminal
        (i.e. the experiment has been executed and is no longer running).
        """
        return self in [ExperimentStatus.DONE, ExperimentStatus.ERROR, ExperimentStatus.CANCELLED]


class ExperimentType(str, Enum):
    QASM = 'QASM'
    PULSE = 'PULSE'


class Experiment(abc.ABC):
    """
    Parent class for a quantum experiment.

    The actual quantum experiment performed on a given quantum processor.
    """

    @abc.abstractmethod
    def query_status(self) -> ExperimentStatus:
        """
        Query the current status of a previously submitted experiment.
        
        If the experiment was already executed successfully when calling this method,
        the results will be automatically populated
        """
        
    @abc.abstractmethod
    def results(self) -> ExperimentResults | None:
        """
        Get the results of a previously submitted experiment (BLOCKING).
        
        If the experiment execution resulted in an error or was previously cancelled, `None` is returned.
        """

    @abc.abstractmethod
    async def wait_for_results(self) -> ExperimentResults | None:
        """
        Wait for the results of a previously submitted experiment (NON-BLOCKING).
        
        If the experiment execution resulted in an error or was previously cancelled, `None` is returned.
        """

    @abc.abstractmethod
    def cancel(self):
        """
        Cancel a previously submitted experiment.
        """

    @abc.abstractmethod
    def as_qasm(self) -> dict:
        """
        Get the Qobj OpenQASM representation of the experiment.
        """
        
    @abc.abstractmethod
    def from_json(self, json: dict) -> Experiment:
        """
        Update an experiment object from a JSON dictionary.
        """

    @abc.abstractmethod
    def _validate(self):
        """
        Validate the experiment in the context of its quantum processor.
        """

    @abc.abstractmethod
    def _initialize(self):
        """
        Initialize the experiment.
        """


class ExperimentResults(abc.ABC):
    """
    Parent class for a quantum experiment results.

    The results of a quantum experiment performed on a given
    quantum processor.
    """

    @property
    @abc.abstractmethod
    def runtime(self) -> float:
        """
        Returns the runtime of the experiment in ns
        (i.e. how long it took for the experiment to run on the given backend)
        """
    
    @abc.abstractmethod
    def from_json(self, json: dict) -> ExperimentResults:
        """
        Initialize an experiment results object from a JSON dictionary.
        """

    @abc.abstractmethod
    def get_counts(self, binary: bool) -> dict:
        """
        Returns the measured counts of the experiment

        Args:
            binary (bool): If True, returns the counts states in binary string format

        Returns:
            dict: A dictionary of states and their respective counts
        """

