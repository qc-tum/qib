from qib.util import const, networking
from qib.circuit import Circuit
from qib.backend import QuantumProcessor, ProcessorConfiguration, ProcessorCredentials, GateProperties, ExperimentStatus
from qib.backend.wmi import WMIOptions, WMIExperiment


class WMIQSimProcessor(QuantumProcessor):
    """
    The WMI Qiskit Simulator quantum processor implementation.
    """

    def __init__(self, access_token: str):
        self.credentials: ProcessorCredentials = ProcessorCredentials(
            url = const.BACK_WMIQSIM_URL,
            access_token = access_token)

    @staticmethod
    def configuration() -> ProcessorConfiguration:
        return ProcessorConfiguration(
            backend_name=const.BACK_WMIQSIM_NAME,
            backend_version=const.BACK_WMIQSIM_VERSION,
            basis_gates=[
                const.GATE_ID,
                const.GATE_X,
                const.GATE_Y,
                const.GATE_H,
                const.GATE_SX,
                const.GATE_RX,
                const.GATE_RY,
                const.GATE_RZ,
                const.GATE_ISWAP,
                const.GATE_CZ
            ],
            conditional=False,
            coupling_map=ProcessorConfiguration.generate_map(3),
            gates=[
                GateProperties(const.GATE_ID, [[0], [1], [2]]),
                GateProperties(const.GATE_X, [[0], [1], [2]]),
                GateProperties(const.GATE_Y, [[0], [1], [2]]),
                GateProperties(const.GATE_H, [[0], [1], [2]]),
                GateProperties(const.GATE_SX, [[0], [1], [2]]),
                GateProperties(const.GATE_RX, [[0], [1], [2]], ['theta']),
                GateProperties(const.GATE_RY, [[0], [1], [2]], ['theta']),
                GateProperties(const.GATE_RZ, [[0], [1], [2]], ['theta']),
                GateProperties(const.GATE_ISWAP, ProcessorConfiguration.generate_map(3)),
                GateProperties(const.GATE_CZ, ProcessorConfiguration.generate_map(3))
            ],
            local=False,
            max_shots=8196,
            meas_level=2,
            memory=True,
            n_qubits=3,
            open_pulse=False,
            simulator=True,
        )

    def submit_experiment(self, name: str, circ: Circuit, options: WMIOptions = WMIOptions()) -> WMIExperiment:
        # experiment
        experiment = WMIExperiment(name, circ, options, self.configuration(), self.credentials)
        
        # request
        response = self._send_request(experiment)
        self._process_response(experiment, response.json())
        return experiment

    def _send_request(self, experiment: WMIExperiment):
        http_headers = {'access-token': self.credentials.access_token, 'Content-Type': 'application/json'}
        return networking.http_put(url = f'{self.credentials.url}/qobj', 
                            headers = http_headers,
                            body = {'qobj': experiment.as_qasm()},
                            title = const.NW_MSG_SEND)
    
    def _process_response(self, experiment: WMIExperiment, response: dict):
        experiment.from_json(response)
        if experiment.status == ExperimentStatus.ERROR:
            raise RuntimeError(f'Experiment could not be submitted: {experiment.error}')
