from qib.util import const
from qib.backend import QuantumProcessor, ProcessorConfiguration


class WMIProcessor(QuantumProcessor):
    """
    The actual WMI Quantum Computer quantum processor implementation.
    """

    def __init__(self):
        self.configuration = ProcessorConfiguration(
            backend_name=const.BACK_WMI_NAME,
            backend_version=const.BACK_WMI_VERSION,
            n_qubits=6,
            basis_gates=[const.GATE_ID, const.GATE_X, const.GATE_Y, const.GATE_SX, const.GATE_RZ],
            local=False,
            simulator=False,
            conditional=False,
            open_pulse=True
        )

    @property
    def configuration(self):
        return self.configuration
