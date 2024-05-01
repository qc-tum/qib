from __future__ import annotations

import abc


class Options(abc.ABC):
    """
    Parent class for a quantum experiment options.

    The options of a quantum experiment performed on a given
    quantum processor.
    """

    @abc.abstractmethod
    def optional(self) -> dict:
        """
        Return a dictionary with the optional parameters and their values (if set).
        """