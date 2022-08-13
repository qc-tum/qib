import abc


class AbstractLattice(abc.ABC):
    """
    Parent class for lattices.
    """

    @property
    @abc.abstractmethod
    def nsites(self) -> int:
        """
        Number of lattice sites.
        """
        pass

    @property
    @abc.abstractmethod
    def ndim(self) -> int:
        """
        Number of spatial dimensions.
        """
        pass

    @abc.abstractmethod
    def adjacency_matrix(self):
        """
        Construct the adjacency matrix, indicating nearest neighbors.
        """
        pass
