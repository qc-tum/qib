import math
import numpy as np
from typing import Sequence
from qib.lattice import AbstractLattice


class CustomizedLattice(AbstractLattice):
    """
    Lattice built from the adjacency matrix.
    """
    def __init__(self, shape: Sequence[int], adj = Sequence[int]):
        self.shape = tuple(shape)
        self.adj = np.array(adj, dtype='bool')
        nsites = math.prod(self.shape)
        if self.adj.shape != (nsites, nsites):
            raise ValueError("The given matrix adj has shape {self.adj.shape} instead of {(nsites,nsites)}.")
        if not np.array_equal(self.adj, self.adj.T):
            raise ValueError("The adjacency matrix must be symmetric.")
        if not all([adj[i,i]==0 for i in range(nsites)]):
            raise ValueError("The adjacency matrix must have a null diagonal")

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
        return self.adj

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
