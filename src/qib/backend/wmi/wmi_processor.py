from qib.backend import QuantumProcessor, ProcessorConfiguration


class WMIProcessor(QuantumProcessor):

    def __init__(self):
        self.configuration = ProcessorConfiguration(
            backend_name="WMIQC",
            backend_version="1.0.0",
            n_qubits=6,
            basis_gates=['id', 'x', 'y', 'sx', 'rz'],
            local=False,
            simulator=False,
            conditional=False,
            open_pulse=True
        )

    @property
    def configuration(self):
        return self.configuration
