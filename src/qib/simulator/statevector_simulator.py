import math
import numpy as np
from typing import Sequence
from qib.simulator import Simulator
from qib.circuit import Circuit
from qib.field import Field


class StatevectorSimulator(Simulator):
    """
    Statevector simulator.
    """

    def run(self, circ: Circuit, description = None):
        """
        Run a quantum circuit simulation.
        """
        fields = circ.fields()
        # assuming initial states is |0,...,0>
        psi = np.zeros(math.prod([f.dof() for f in fields]))
        psi[0] = 1
        # apply gates
        for g in circ.gates:
            # TODO: matrix-free application of gate
            psi = g.as_circuit_matrix(fields) @ psi
        return psi
