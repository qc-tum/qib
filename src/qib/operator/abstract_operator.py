import abc


class AbstractOperator(abc.ABC):
    """
    Abstract parent class for operators.
    """

    @abc.abstractmethod
    def is_unitary(self):
        """
        Whether the operator is unitary.
        """
        pass

    @abc.abstractmethod
    def is_hermitian(self):
        """
        Whether the operator is Hermitian.
        """
        pass

    @abc.abstractmethod
    def as_matrix(self):
        """
        Generate the (sparse) matrix representation of the operator.
        """
        pass
