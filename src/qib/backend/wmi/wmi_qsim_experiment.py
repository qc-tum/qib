from __future__ import annotations

from typing import Any
from itertools import combinations

from qib.util import const
from qib.field import Particle
from qib.circuit import Circuit
from qib.backend import Experiment, ExperimentResults, ExperimentType, ProcessorConfiguration
from qib.backend.wmi import WMIQSimOptions


class WMIQSimExperiment(Experiment):
    """
    The WMI Qiskit Simulator quantum experiment implementation.
    """
    
    def __init__(self,
                 name: str, 
                 circuit: Circuit,
                 options: WMIQSimOptions,
                 configuration: ProcessorConfiguration,
                 type: ExperimentType = ExperimentType.QASM):
        super().__init__(name, circuit, options, configuration, type)
        self.qobj_id = const.QOBJ_ID_QISM_EXPERIMENT
        self.schema_version = const.QOBJ_SCHEMA_VERSION
        self._validate()
        
    def query_status(self) -> WMIQSimExperimentResults:
        # TODO: Ensure that experiment was submitted (status != INITIALIZING)
        pass
    
    async def wait_for_results(self) -> WMIQSimExperimentResults:
        # TODO: Ensure that experiment was submitted (status != INITIALIZING)
        pass
    
    def cancel(self) -> WMIQSimExperimentResults:
        pass
    
    def as_openQASM(self) -> dict:
        qubits: set[Particle] = self.circuit.particles()
        clbits: set[int] = self.circuit.clbits()
        return {
            'qobj_id': self.qobj_id,
            'type': self.type.value,
            'schema_version': self.schema_version,
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
                'backend_name': self.configuration.backend_name,
                'backend_version': self.configuration.backend_version,
                },
            'config': {
                'shots': self.options.shots,
                'memory': self.configuration.memory,
                'meas_level': self.configuration.meas_level,
                'init_qubits': self.options.init_qubits,
                'do_emulation': False,
                'memory_slots': len(clbits),
                'n_qubits': len(qubits)
            },
        }
        
    def _validate(self):
        # check that the number of shots is not exceeded
        if self.options.shots > self.configuration.max_shots:
            raise ValueError("Number of shots exceeds maximum allowed number of shots.")

        for gate in self.circuit.gates:
            gate_openQASM = gate.as_openQASM()
            gate_name = gate_openQASM['name']
            gate_qubits = gate_openQASM['qubits']
            gate_params = gate_openQASM['params'] if 'params' in gate_openQASM else []
            
            if gate_name != 'measure':
                # check that the gate is supported by the processor
                if gate_name not in self.configuration.basis_gates:
                    raise ValueError(f"Gate {type(gate)} is not supported by the processor.")
                
                # check that the used qubits are configured for the gate
                gate_properties = self.configuration.get_gate_by_name(gate_name)
                if gate_properties is None:
                    raise ValueError(f"Gate {type(gate)} is not configured by the processor.")
                if not gate_properties.check_qubits(gate_qubits):
                    raise ValueError(f"Gate {type(gate)} is not configured for the used qubits.")
                if not gate_properties.check_params(gate_params):
                    raise ValueError(f"Gate {type(gate)} is not configured for the used parameters.")

            # check that gates are performed only on coupled qubits
            if len(gate_qubits) > 1 and self.configuration.coupling_map:
                qubit_pairs = list(combinations(gate_qubits, 2))
                for qubit_pair in qubit_pairs:
                    if qubit_pair not in self.configuration.coupling_map:
                        raise ValueError(f"Gate {type(gate)} is not performed on coupled qubits.")
            
        # check that the number of qubits is adequate
        qubits: set[Particle] = gate.particles()
        qubits_index = [q.index for q in qubits]
        if len(qubits) > self.configuration.n_qubits \
        or min(qubits_index) < 0 \
        or max(qubits_index) >= self.configuration.n_qubits:
            raise ValueError("Number of qubits exceeds maximum allowed number of qubits, or indexes are incorrect.")


class WMIQSimExperimentResults(ExperimentResults):
    pass
