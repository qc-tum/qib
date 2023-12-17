from typing import Sequence
from qib.backend import QuantumProcessor
from qib.backend import ProcessorConfiguration, ProcessorOptions
from qib.circuit import Circuit
from qib.field import Field


class QiskitSimProcessor(QuantumProcessor):

    def __init__(self):
        self.configuration = ProcessorConfiguration(
            backend_name="QiskitSimulator",
            backend_version="1.0.0",
            n_qubits=3,
            basis_gates=['x', 'sx', 'rz', 'cz'],
            local=False,
            simulator=True,
            conditional=False,
            open_pulse=False
        )

        self._default_options = QiskitSimOptions(
            shots=1024,
            memory=False,
            do_emulation=True
        )

    @property
    def configuration(self):
        return self.configuration

    def submit(self, circ: Circuit, fields: Sequence[Field], description):
        # TODO: transpile circuit into QObj
        # TODO: send QObj to WMI Backend
        # TODO: Register & return quantum process with WMI Backend
        pass

    def query_results(self, experiment):
        pass


class QiskitSimOptions(ProcessorOptions):

    def __init__(self, shots, memory, do_emulation):
        super().__init__(shots, memory, do_emulation)
