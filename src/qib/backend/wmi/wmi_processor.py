from typing import Sequence
from qib.backend import QuantumProcessor, ProcessorConfiguration
from qib.backend.wmi import WMIOptions
from qib.circuit import Circuit
from qib.field import Field


class WMIProcessor(QuantumProcessor):

    def __init__(self):
        self.configuration = ProcessorConfiguration(
            backend_name="WMIQC",
            backend_version="1.0.0",
            n_qubits=6,
            basis_gates=['id', 'x', 'y', 'sx', 'rz', 'cz'],
            local=False,
            simulator=False,
            conditional=False,
            open_pulse=True
        )

    @property
    def configuration(self):
        return self.configuration

    def submit(self, circ: Circuit, fields: Sequence[Field], description):
        pass

    def query_results(self, experiment):
        pass
