from __future__ import annotations

import abc
from typing import Sequence
from qib.circuit import Circuit
from qib.field import Field
from qib.backend.experiment import Experiment
from qib.backend.options import Options


class QuantumProcessor(abc.ABC):
    """
    Parent class for quantum processor (a.k.a. backend).

    A quantum processor is a device that can execute quantum circuits,
    be it a quantum computer, a simulator, or a different type of quantum backend.
    """

    @staticmethod
    @abc.abstractmethod
    def configuration() -> ProcessorConfiguration:
        """
        The configuration of the quantum processor.
        """

    @abc.abstractmethod
    def submit_experiment(self, name: str, circ: Circuit, options: Options) -> Experiment:
        """
        Submit a quantum circuit and experiment execution options to a quantum processor backend,
        returning a validated "experiment" object to query the results.
        """

    @abc.abstractmethod
    def _validate_experiment(self, circ: Circuit, options: Options):
        """
        Validate resulting experiment in the context of this quantum processor.
        """

    @abc.abstractmethod
    def _send_experiment(self, experiment: Experiment):
        """
        Send resulted experiment to the quantum processor backend.
        """


class ProcessorConfiguration:
    """
    Generic class for quantum processor configuration.

    The configuration of a quantum processor includes information about the processor itself,
    such as the number of qubits, the available gates, the coupling map, etc.
    """

    def __init__(
            self,
            backend_name: str,
            backend_version: str,
            basis_gates: Sequence[str],
            conditional: bool,
            coupling_map: Sequence[Sequence[int]],
            gates: Sequence[GateProperties],
            local: bool,
            max_shots: int,
            meas_level: int,
            memory: bool,
            n_qubits: int,
            open_pulse: bool,
            simulator: bool,
    ) -> None:
        self.backend_name: str = backend_name
        self.backend_version: str = backend_version
        self.basis_gates: Sequence[str] = basis_gates
        self.conditional: bool = conditional
        self.coupling_map: Sequence[Sequence[int]] = coupling_map
        self.gates: Sequence[GateProperties] = gates
        self.local: bool = local
        self.max_shots: int = max_shots
        self.meas_level: int = meas_level
        self.memory: bool = memory 
        self.n_qubits: int = n_qubits
        self.open_pulse: bool = open_pulse
        self.simulator: bool = simulator

    def get_gate_by_name(self, gate_name: str)-> GateProperties:
        """
        Get a gate properties by its name.
        """
        for gate in self.gates:
            if gate.name == gate_name:
                return gate
        return None

class GateProperties:
    """
    Generic class for gate properties.

    The properties of a quantum processor's gate, including information about what
    gates are configured for which qubits of the targeted quantum system.
    """
    def __init__(
            self,
            name: str,
            qubits: Sequence[Sequence[int]],
            parameters: Sequence[str] = []
    ) -> None:
        self.name: str = name
        self.qubits: Sequence[int] = qubits
        self.parameters: Sequence[str] = parameters

    def check_qubits(self, qubits: Sequence[int]) -> bool:
        """
        Check if the gate is configured for the given qubits.
        """
        return qubits in self.qubits
    
    def check_params(self, params: Sequence) -> bool:
        """
        Check if the gate is configured for the given parameters.
        """
        return len(params) == len(self.parameters)