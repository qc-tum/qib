import abc


class AbstractOperator(abc.ABC):
    """
    Abstract parent class for operators.
    """

    @abc.abstractmethod
    def as_matrix(self):
        """
        Generate the (sparse) matrix representation of the operator.
        """
        pass
