from __future__ import annotations

from qib.backend.options import Options


class QiskitSimOptions(Options):

    def __init__(self,
                 shots,
                 memory,
                 do_emulation
                 ):
        super().__init__(shots, memory, do_emulation)

    @staticmethod
    def default() -> QiskitSimOptions:
        return QiskitSimOptions(
            shots=1024,
            memory=False,
            do_emulation=True
        )
