from __future__ import annotations

import abc


class Options(abc.ABC):
    """
    Parent class for a quantum experiment options.

    The options of a quantum experiment performed on a given
    quantum processor.
    """

    @abc.abstractmethod
    def __init__(
            self,
            shots: int,
            init_qubits: bool,
    ) -> None:
        self.shots: int = shots
        self.init_qubits: bool = init_qubits

    @staticmethod
    @abc.abstractmethod
    def default() -> Options:
        """
        The default options used for an experiment on a given processor,
        if no custom options have been specified.
        """
