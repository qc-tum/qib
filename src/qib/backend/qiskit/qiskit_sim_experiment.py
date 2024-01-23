from typing import Any

from qib.util import const
from qib.field import Particle
from qib.circuit import Circuit
from qib.backend import Experiment, ExperimentResults, ExperimentType, Options
from qib.backend.qiskit import QiskitSimOptions, QiskitSimExperiment, QiskitSimExperimentResults


class QiskitSimExperiment(Experiment):
    """
    The WMI Qiskit Simulator quantum experiment implementation.
    """
    
    def __init__(self, name:str, circuit: Circuit, options: QiskitSimOptions, type: ExperimentType = ExperimentType.QASM):
        super().__init__(name, circuit, options, type)
        self.qobj_id = const.QOBJ_ID_QISM_EXPERIMENT
        
    def query_status(self) -> QiskitSimExperimentResults:
        # TODO: Ensure that experiment was submitted (status != INITIALIZING)
        pass
    
    async def wait_for_results(self) -> QiskitSimExperimentResults:
        # TODO: Ensure that experiment was submitted (status != INITIALIZING)
        pass
    
    def cancel(self) -> QiskitSimExperimentResults:
        pass
    
    def as_openQASM(self) -> dict:
        qubits: set[Particle] = self.circuit.particles()
        clbits: set[int] = self.circuit.clbits()
        return {
            'qobj_id': self.qobj_id,
            'type': self.type.value,
            'schema_version': const.QOBJ_SCHEMA_VERSION,
            'experiments': [
                    {
                        'header': {
                            'qubit_labels': {'qubits': [['q', qubit.index] for qubit in qubits]},
                            'n_qubits': len(qubits),
                            'qreg_sizes': len(qubits),
                            'clbit_labels': {'clbits': [['c', clbit] for clbit in clbits]},
                            'memory_slots': len(clbits),
                            'creg_sizes': len(clbits),
                            'name': self.name,
                            'global_phase': 0.0,
                            'metadata': {}
                            },
                        'config': {
                            'n_qubits': len(qubits),
                            'memory_slots': len(clbits)
                            },
                        'instructions': self.instructions
                    }
                ],
            'header': {
                'backend_name': const.BACK_WMIQSIM_NAME,
                'backend_version': const.BACK_WMIQSIM_VERSION
                },
            'config': {
                'shots': self.options.shots,
                'memory': True,
                'meas_level': 2,
                'init_qubits': self.options.init_qubits,
                'do_emulation': False,
                'memory_slots': len(clbits),
                'n_qubits': len(qubits)
            },
        }


class QiskitSimExperimentResults(ExperimentResults):
    pass
