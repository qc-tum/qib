from __future__ import annotations

from itertools import combinations
import time, sched, asyncio
import uuid

from qib.util import const, networking
from qib.field import Particle
from qib.circuit import Circuit
from qib.backend import ExperimentStatus, Experiment, ExperimentResults, ExperimentType, ProcessorConfiguration, ProcessorCredentials
from qib.backend.wmi import WMIOptions


class WMIExperiment(Experiment):
    """
    The WMI quantum experiment implementation.
    """
    
    def __init__(self,
                 name: str, 
                 circuit: Circuit,
                 options: WMIOptions,
                 configuration: ProcessorConfiguration,
                 credentials: ProcessorCredentials,
                 type: ExperimentType = ExperimentType.QASM):
        self.name: str = name
        self.circuit: Circuit = circuit
        self.options: WMIOptions = options
        self.type: ExperimentType = type
        self.configuration: ProcessorConfiguration = configuration
        self.credentials: ProcessorCredentials = credentials
        self._initialize()
        self._validate()
    
    @property
    def job_id(self) -> str:
        return self._job_id
    
    def query_status(self) -> ExperimentStatus:
        # check current status
        if self.status == ExperimentStatus.INITIALIZING:
            raise ValueError("Experiment has to be submitted first.")
        elif (self.status.is_terminal()):
            return self.status
        
        # query incoming status
        http_headers = {'access-token': self.credentials.access_token, 'Content-Type': 'application/json'}
        response = networking.http_post(url = f'{self.credentials.url}/qobj',
                                      headers = http_headers,
                                      body = {'job_id': self._job_id},
                                      title = const.NW_MSG_QUERY)
        self.from_json(response.json())
        
        # update results
        if self.status == ExperimentStatus.DONE:
            self._results = WMIExperimentResults(self).from_json(response.json())

        return self.status

    def results(self) -> WMIExperimentResults | None:
        if self._results is not None and self.status is ExperimentStatus.DONE: return self._results
        
        scheduler = sched.scheduler(time.time, time.sleep)
        def check_and_reschedule(scheduler):
            if not self.query_status().is_terminal():
                scheduler.enter(const.NW_QUERY_FRQ, 1, check_and_reschedule, (scheduler,))
        scheduler.enter(0, 1, check_and_reschedule, (scheduler,))
        scheduler.run()
        
        if self.status is ExperimentStatus.DONE: return self._results
        return None

    async def wait_for_results(self) -> WMIExperimentResults | None:
        if self._results is not None and self.status is ExperimentStatus.DONE: return self._results
        
        while not self.query_status().is_terminal():
            await asyncio.sleep(const.NW_QUERY_FRQ)
            
        if self.status is ExperimentStatus.DONE: return self._results
        return None
    
    def cancel(self):
        # TODO: cancel running experiment
        raise NotImplementedError("Cancelling experiments is not yet supported for WMI backends.")
    
    def as_openQASM(self) -> dict:
        qubits: set[Particle] = self.circuit.particles()
        clbits: set[int] = self.circuit.clbits()
        qobj = {
            'qobj_id': str(self.qobj_id),
            'type': self.type.value,
            'schema_version': self.schema_version,
            'experiments': [
                    {
                        'header': {
                            'qubit_labels': {'qubits': [['q', qubit.index] for qubit in qubits]},
                            'n_qubits': len(qubits),
                            'qreg_sizes': {'q': len(qubits)},
                            'clbit_labels': {'clbits': [['c', clbit] for clbit in clbits]},
                            'memory_slots': len(clbits),
                            'creg_sizes': {'q': len(clbits)},
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
                # Required properties
                'shots': self.options.shots,
                'memory': self.configuration.memory,
                'meas_level': self.configuration.meas_level,
                'init_qubits': self.options.init_qubits,
                'do_emulation': self.options.do_emulation,
                'memory_slots': len(clbits),
                'n_qubits': len(qubits),
            },
        }
        
        # Optional properties
        qobj['config'].update(self.options.optional())
        
        return qobj
        
    def from_json(self, json: dict) -> WMIExperiment:
        self._job_id = json['job_id']
        self._execution_datetime = json['execution_datetime']
        self._from_wmi_status(json['status'])
        return self
        
    def _initialize(self):
        self.error: str = None
        self.instructions: list = self.circuit.as_openQASM()
        self.status: ExperimentStatus = ExperimentStatus.INITIALIZING
        
        self.qobj_id: uuid.UUID = uuid.uuid4()
        self.schema_version: str = const.QOBJ_SCHEMA_VERSION
        
        self._job_id: str = None
        self._execution_datetime: str = None
        self._results: WMIExperimentResults  = None
        
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
                    raise ValueError(f"Gate {gate_name.upper()} ({type(gate)}) is not supported by the processor.")
                
                # check that the used qubits are configured for the gate
                gate_properties = self.configuration.get_gate_by_name(gate_name)
                if gate_properties is None:
                    raise ValueError(f"Gate {gate_name.upper()} {type(gate)} is not configured by the processor.")
                if not gate_properties.check_qubits(gate_qubits):
                    raise ValueError(f"Gate {gate_name.upper()} {type(gate)} is not configured for the used qubits.")
                if not gate_properties.check_params(gate_params):
                    raise ValueError(f"Gate {gate_name.upper()} {type(gate)} is not configured for the used parameters.")

                # check that gates are performed only on coupled qubits
                if len(gate_qubits) > 1 and self.configuration.coupling_map:
                    qubit_pairs = list(combinations(gate_qubits, 2))
                    for qubit_pair in qubit_pairs:
                        if list(qubit_pair) not in self.configuration.coupling_map:
                            raise ValueError(f"Gate {gate_name.upper()} {type(gate)} is not performed on coupled qubits.")
            
        # check that the number of qubits is adequate
        qubits: set[Particle] = gate.particles()
        qubits_index = [q.index for q in qubits]
        if len(qubits) > self.configuration.n_qubits \
        or min(qubits_index) < 0 \
        or max(qubits_index) >= self.configuration.n_qubits:
            raise ValueError("Number of qubits exceeds maximum allowed number of qubits, or indexes are incorrect.")

    def _from_wmi_status(self, status: str):
        """
        Set the status of the experiment (convert from WMI-specific format).
        """
        if status == 'pending':
            self.status = ExperimentStatus.QUEUED
        elif status == 'active':
            self.status = ExperimentStatus.RUNNING
        elif status == 'finished':
            self.status = ExperimentStatus.DONE
        elif status == 'cancelled':
            self.status = ExperimentStatus.CANCELLED
        elif status == 'offline':
            self.status = ExperimentStatus.ERROR
            self.error = 'The backend is offline.'
        else:
            self.status = ExperimentStatus.ERROR
            self.error = 'Unknown error.'


class WMIExperimentResults(ExperimentResults):
    """
    The WMI quantum experiment results implementation.
    """
    def __init__(self, experiment: WMIExperiment):
        self._experiment_ref: WMIExperiment = experiment

    @property
    def runtime(self) -> float:
        return self._runtime
        
    def from_json(self, json: dict) -> WMIExperimentResults:
        self._runtime: float = json['runtime']
        self._counts: dict = json['counts'][0]
        return self
    
    def get_counts(self, binary: bool = False) -> dict:
        if binary:
            n_qubits = len(self._experiment_ref.circuit.particles())
            return {str(bin(int(key, 16))).split('b')[1].zfill(n_qubits): 
                value for key, value in self._counts.items()}
        return self._counts
    
    def plot_histogram(self):
        # TODO: matplotlib histogram plotting of counts. 
        # Do we want to have matplotlib as a dependency? (Qiskit: matplotlib and latex)
        raise NotImplementedError("Plotting histograms is not yet supported.")
