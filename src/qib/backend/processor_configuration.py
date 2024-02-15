from __future__ import annotations

from typing import Sequence

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
    ):
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
    
    @staticmethod
    def coupling_map(n_qubits: int) -> Sequence[Sequence[int]]:
        """
        Generate a fully-coupled coupling map for a given number of qubits.
        """
        return [[q1, q2] for q1 in range(n_qubits) for q2 in range(n_qubits) if q1 != q2]


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
    ):
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


class ProcessorCredentials:
    """
    Generic class for quantum processor credentials.
    
    Includes parameters on how to access a quantum processor backend.
    """

    def __init__(self,
                 url: str,
                 access_token: str):
        self.url: str = url
        self.access_token: str = access_token