import numpy as np
from typing import Sequence
from qib.operator import AbstractOperator, PauliOperator
from qib.vqe.ansatz import Ansatz
from qib.vqe.optimizer import Optimizer
from scipy.optimize import minimize, OptimizeResult


def measure_expectation_statevector(pauli_op: PauliOperator, state: Sequence[float]):
    """
    Given a Pauli operator and a quantum state, it calculates the expectation value.
    """
    state = np.array(state, copy=False)
    return (state.T@pauli_op.as_matrix().toarray())@state


# TODO: add more measurement methods
class VQE:
    """
    VQE algorithm.
    """
    def __init__(self, ansatz: Ansatz, optimizer: Optimizer, initial_state: Sequence[float], measure_method: str="statevector"):
        self.ansatz = ansatz
        self.optimizer = optimizer
        if optimizer.x0 is None:
            self.optimizer.x0 = np.random.rand(self.ansatz.num_parameters)
        self.initial_state = np.array(initial_state)
        if not measure_method == "statevector":
            raise NotImplementedError(f"The measuring method {measure_method} has not been implemented yet. Only 'statevector' is available.")
        self.measure_method = measure_method
        self._optimal_params = None

    def run(self, pauli_op: PauliOperator) -> OptimizeResult:
        def energy_func(params):
            # starts form _initial_state and applies ansatz.
            state = self.ansatz.as_matrix(params).toarray()@self.initial_state
            return measure_expectation_statevector(pauli_op, state)
            
        res = minimize(fun = energy_func,
                       x0 = self.optimizer.x0,
                       args = self.optimizer.args,
                       method = self.optimizer.method,
                       jac = self.optimizer.jac,
                       tol = self.optimizer.tol,
                       callback = self.optimizer.callback,
                       options = self.optimizer.options)
        
        self._optimal_params = res.x
        return res

    def expectation_secondary_ops(self, secondary_ops: Sequence[PauliOperator]):
        if self._optimal_params is None:
            return None
        else:
            state = self.ansatz.as_matrix(params).toarray()@self.initial_state
            return [measure_expectation_statevector(s_op, state) for s_op in secondary_ops]
