from __future__ import annotations

import abc


class Options(abc.ABC):

    @abc.abstractmethod
    def __init__(
            self,
            shots: int,
    ) -> None:
        self.shots: int = shots

    @staticmethod
    @abc.abstractmethod
    def default() -> Options:
        pass
