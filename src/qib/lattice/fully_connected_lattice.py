import math
import numpy as np
from typing import Sequence, Union
from qib.lattice import AbstractLattice


class FullyConnectedLattice(AbstractLattice):
    """
    Fully connected lattice of n elements.
    Can be used to represent orbitals in a molecule
    """
    def __init__(self, shape: Union[int, Sequence[int]]):
        if type(shape) == int:
            self.shape = (shape,)
        else:
            self.shape = tuple(shape)
            
    @property
    def nsites(self) -> int:
        """
        Number of lattice sites.
        """
        return math.prod(self.shape)

    @property
    def ndim(self) -> int:
        """
        Number of spatial dimensions.
        """
        return len(self.shape)

    def adjacency_matrix(self):
        """
        Construct the adjacency matrix, indicating nearest neighbors.
        """
        adj = np.ones((self.nsites, self.nsites), dtype=int)
        for n in range(self.nsites):
            adj[n,n] = 0;
        return adj

    def index_to_coord(self, i: int) -> tuple:
        """
        Map linear index to lattice coordinate.
        """
        assert i < self.nsites
        return np.unravel_index(i, self.shape)

    def coord_to_index(self, c) -> int:
        """
        Map lattice coordinate to linear index.
        """
        for i in range(len(self.shape)):
            assert c[i] < self.shape[i]
        return int(np.ravel_multi_index(c, self.shape))
