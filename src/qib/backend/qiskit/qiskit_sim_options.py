from __future__ import annotations

from qib.backend.options import Options


class QiskitSimOptions(Options):

    def __init__(self,
                 shots
                 ):
        super().__init__(shots)

    @staticmethod
    def default() -> QiskitSimOptions:
        return QiskitSimOptions(
            shots=1024
        )
