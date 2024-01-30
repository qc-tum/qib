from qib.util import const
from qib.circuit import Circuit
from qib.backend import QuantumProcessor, ProcessorConfiguration, GateProperties
from qib.backend.wmi import WMIQSimOptions, WMIQSimExperiment


class WMIQSimProcessor(QuantumProcessor):
    """
    The WMI Qiskit Simulator quantum processor implementation.
    """

    def __init__(self, access_token: str):
        self.url: str = const.BACK_WMIQSIM_URL
        self.access_token: str = access_token

    @staticmethod
    def configuration() -> ProcessorConfiguration:
        return ProcessorConfiguration(
            backend_name=const.BACK_WMIQSIM_NAME,
            backend_version=const.BACK_WMIQSIM_VERSION,
            basis_gates=[const.GATE_X, const.GATE_SX, const.GATE_RZ, const.GATE_CZ],
            conditional=False,
            coupling_map=None,
            gates=[
                GateProperties(const.GATE_X, [[0]]),
                GateProperties(const.GATE_SX, [[0]]),
                GateProperties(const.GATE_RZ, [[0]], ['theta']),
                GateProperties(const.GATE_CZ, [[1,0], [2,0]])
            ],
            local=False,
            max_shots=8196,
            meas_level=2,
            memory=True,
            n_qubits=3,
            open_pulse=False,
            simulator=True,
        )

    def submit_experiment(self, name: str, circ: Circuit, options: WMIQSimOptions = WMIQSimOptions.default()) -> WMIQSimExperiment:
        experiment = WMIQSimExperiment(name, circ, options, self.configuration())
        self._send_experiment(experiment)
        return experiment

    def _send_experiment(self, experiment: WMIQSimExperiment):
        # TODO: build HTTP request
        # TODO: send HTTP request via the HTTP PUT endpoint
        pass
