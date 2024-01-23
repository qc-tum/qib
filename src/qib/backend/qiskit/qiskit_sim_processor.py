from itertools import combinations

from qib.util import const
from qib.circuit import Circuit
from qib.field import Particle
from qib.backend import QuantumProcessor, ProcessorConfiguration, GateProperties
from qib.backend.qiskit import QiskitSimExperiment, QiskitSimOptions


class QiskitSimProcessor(QuantumProcessor):
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
            n_qubits=3,
            basis_gates=[const.GATE_X, const.GATE_SX, const.GATE_RZ, const.GATE_CZ],
            gates=[
                GateProperties(const.GATE_X, [[0]]),
                GateProperties(const.GATE_SX, [[0]]),
                GateProperties(const.GATE_RZ, [[0]], ['theta']),
                GateProperties(const.GATE_CZ, [[1,0], [2,0]])
            ],
            coupling_map=None,
            local=False,
            simulator=True,
            conditional=False,
            open_pulse=False
        )

    def submit_experiment(self, name: str, circ: Circuit, options: QiskitSimOptions = QiskitSimOptions.default()) -> QiskitSimExperiment:
        self._validate_experiment(circ, options)
        experiment = QiskitSimExperiment(name, circ, options)
        self._send_experiment(experiment)
        return experiment

    def _validate_experiment(self, circ: Circuit, options: QiskitSimOptions):
        # check that the number of shots is not exceeded
        if options.shots > self.configuration().max_shots:
            raise ValueError("Number of shots exceeds maximum allowed number of shots.")

        for gate in circ.gates:
            gate_openQASM = gate.as_openQASM()
            gate_name = gate_openQASM['name']
            gate_qubits = gate_openQASM['qubits']
            gate_params = gate_openQASM['params'] if 'params' in gate_openQASM else []
            
            if gate_name != 'measure':
                # check that the gate is supported by the processor
                if gate_name not in self.configuration().basis_gates:
                    raise ValueError(f"Gate {type(gate)} is not supported by the processor.")
                
                # check that the used qubits are configured for the gate
                gate_properties = self.configuration().get_gate_by_name(gate_name)
                if gate_properties is None:
                    raise ValueError(f"Gate {type(gate)} is not configured by the processor.")
                if not gate_properties.check_qubits(gate_qubits):
                    raise ValueError(f"Gate {type(gate)} is not configured for the used qubits.")
                if not gate_properties.check_params(gate_params):
                    raise ValueError(f"Gate {type(gate)} is not configured for the used parameters.")

            # check that gates are performed only on coupled qubits
            if len(gate_qubits) > 1 and self.configuration().coupling_map:
                qubit_pairs = list(combinations(gate_qubits, 2))
                for qubit_pair in qubit_pairs:
                    if qubit_pair not in self.configuration().coupling_map:
                        raise ValueError(f"Gate {type(gate)} is not performed on coupled qubits.")
            
        # check that the number of qubits is adequate
        qubits: set[Particle] = gate.particles()
        qubits_index = [q.index for q in qubits]
        if len(qubits) > self.configuration().n_qubits \
        or min(qubits_index) < 0 \
        or max(qubits_index) >= self.configuration().n_qubits:
            raise ValueError("Number of qubits exceeds maximum allowed number of qubits, or indexes are incorrect.")

    def _send_experiment(self, experiment: QiskitSimExperiment):
        # TODO: build HTTP request
        # TODO: send HTTP request via the HTTP PUT endpoint
        pass
