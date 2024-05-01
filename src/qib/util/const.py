# Gates, operators, and instructions OpenQASM standard names
GATE_CH: str = 'ch'
GATE_CR: str = 'cu3'
GATE_CRX: str = 'crx'
GATE_CRY: str = 'cry'
GATE_CRZ: str = 'crz'
GATE_CS: str = 'cs'
GATE_CSDG: str = 'csdg'
GATE_CX: str = 'cx'
GATE_CY: str = 'cy'
GATE_CZ: str = 'cz'
GATE_H: str = 'h'
GATE_ID: str = 'id'
GATE_ISWAP: str = 'iswap'
GATE_R: str = 'u3'
GATE_RX: str = 'rx'
GATE_RY: str = 'ry'
GATE_RZ: str = 'rz'
GATE_S: str = 's'
GATE_SDG: str = 'sdg'
GATE_SX: str = 'sx'
GATE_T: str = 't'
GATE_TDG: str = 'tdg'
GATE_TOFFOLI: str = 'ccx'
GATE_X: str = 'x'
GATE_Y: str = 'y'
GATE_Z: str = 'z'
INSTR_MEASURE: str = 'measure'
INSTR_BARRIER: str = 'barrier'
INSTR_DELAY: str = 'delay'

# Backend quantum processor's parameters and constants
BACK_WMIQSIM_URL = 'https://wmiqc-api.wmi.badw.de/1/qiskitSimulator'
BACK_WMIQSIM_NAME = 'dedicatedSimulator'
BACK_WMIQSIM_VERSION = '1.0.0'
BACK_WMIQC_URL = 'https://wmiqc-api.wmi.badw.de/1/wmiqc'
BACK_WMIQC_NAME = 'dedicated'
BACK_WMIQC_VERSION = '1.0.0'

# Qobj parameters and constants
QOBJ_SCHEMA_VERSION = '1.3.0'

# Networking parameters and constants
NW_TIMEOUT: int = 10  # seconds
NW_MAX_RETRIES: int = 5
NW_MSG_SEND: str = 'SUBMIT EXPERIMENT'
NW_MSG_QUERY: str = 'QUERY EXPERIMENT'
NW_QUERY_FRQ_SLOW: float = 2  # seconds
NW_QUERY_FRQ_FAST: float = 0.2  # seconds
