from __future__ import annotations

from qib.backend.options import Options


class WMIQSimOptions(Options):
    """
    The WMI Qiskit Simulator quantum experiment options.
    """

    def __init__(self,
                 shots,
                 init_qubits,
                 do_emulation
                 ):
        super().__init__(shots, init_qubits)
        self.do_emulation = do_emulation

    @staticmethod
    def default() -> WMIQSimOptions:
        return WMIQSimOptions(
            shots=1024,
            init_qubits=True,
            do_emulation=False
        )
