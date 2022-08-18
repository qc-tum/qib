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

    @abc.abstractmethod
    def index_to_coord(self, i: int) -> tuple:
        """
        Map linear index to lattice coordinate.
        """
        pass

    @abc.abstractmethod
    def coord_to_index(self, c) -> int:
        """
        Map lattice coordinate to linear index.
        """
        pass
