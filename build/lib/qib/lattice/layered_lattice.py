import numpy as np
from qib.lattice import AbstractLattice


class LayeredLattice(AbstractLattice):
    """
    Layered lattice, consisting of connected layers of the same base lattice.
    """
    def __init__(self, base_lattice: AbstractLattice, nlayers: int):
        if nlayers < 1:
            raise ValueError(f"number of layers must be a positive integer, received {nlayers}")
        self.nlayers = int(nlayers)
        self.base_lattice = base_lattice

    @property
    def nsites(self) -> int:
        """
        Number of lattice sites.
        """
        return self.nlayers * self.base_lattice.nsites

    @property
    def ndim(self) -> int:
        """
        Number of spatial dimensions.
        """
        return 1 + self.base_lattice.ndim

    def adjacency_matrix(self):
        """
        Construct the adjacency matrix, indicating nearest neighbors.
        """
        # a site is connected to all peer sites in the other layers
        base_adj = self.base_lattice.adjacency_matrix()
        layneigh = np.identity(self.base_lattice.nsites, dtype=int)
        return np.block([i*[layneigh] + [base_adj] + (self.nlayers-i-1)*[layneigh] for i in range(self.nlayers)])

    def index_to_coord(self, i: int) -> tuple:
        """
        Map linear index to lattice coordinate.
        """
        assert i < self.nsites
        # additional coordinate for layers
        return (i // self.base_lattice.nsites,) + self.base_lattice.index_to_coord(i % self.base_lattice.nsites)

    def coord_to_index(self, c) -> int:
        """
        Map lattice coordinate to linear index.
        """
        assert c[0] < self.nlayers
        return c[0] * self.base_lattice.nsites + self.base_lattice.coord_to_index(c[1:])
