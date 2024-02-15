from __future__ import annotations

from typing import Sequence

from qib.backend.options import Options


class WMIOptions(Options):
    """
    The WMI Qiskit Simulator quantum experiment options.
    """

    def __init__(self,
                 shots: int = 1024,
                 init_qubits: bool = True,
                 do_emulation: bool = False,
                 loops: dict = {},
                 sequence_settings: dict = {},
                 reference_measurement: dict = {'function': 'nothing'},
                 trigger_time: float = 0.001,
                 relax: bool = True,
                 relax_time: int = None,
                 default_qubits: Sequence[str] = None,
                 weighting_amp: float = 1.0,
                 acquisition_mode: str = 'integration_trigger',
                 averaging_mode: str = 'single_shot',
                 log_level: str = 'debug',
                 log_level_std: str = 'info',
                 log_file_level: str = 'debug',
                 store_nt_result: bool = True,
                 name_suffix: str = '',
                 fridge: str = 'badwwmi-021-xld105',
                 chip: str = 'dedicated'
                 ):
        super().__init__(shots, init_qubits)
        self.do_emulation: bool = do_emulation
        self.loops: dict = loops
        self.sequence_settings: dict = sequence_settings
        self.reference_measurement: dict = reference_measurement
        self.trigger_time: float = trigger_time
        self.relax: bool = relax
        self.relax_time: int = relax_time
        self.default_qubits: Sequence[str] = default_qubits
        self.weighting_amp: float = weighting_amp
        self.aquisition_mode: str = acquisition_mode
        self.average_mode: str = averaging_mode
        self.log_level: str = log_level
        self.log_level_std: str = log_level_std
        self.log_file_level: str = log_file_level
        self.store_nt_result: bool = store_nt_result
        self.name_suffix: str = name_suffix
        self.fridge: str = fridge
        self.chip: str = chip

    @staticmethod
    def default() -> WMIOptions:
        return WMIOptions()
