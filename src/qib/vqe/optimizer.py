import numpy as np


class Optimizer:
    """
    Optimizer class.
    Holds the arguments for initializing 'minimize' function in SciPy.
    Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
    """
    def __init__(self, x0=None, args=(), method="COBYLA", jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        if x0 is None:
            self.x0 = None
        else:
            self.x0 = np.array(x0)
        self.args = args
        self.method = method
        self.jac = jac
        self.hess = hess
        self.hessp = hessp
        self.bounds = bounds
        self.constraints = constraints
        self.tol = tol
        self.callback = callback
        self.options = options
