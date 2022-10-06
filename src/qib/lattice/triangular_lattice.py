import math
import numpy as np
from typing import Sequence
from qib.lattice import AbstractLattice


class TriangularLattice(AbstractLattice):
    """
    Triangular lattice.
    Formally equal to an integer lattice with one chord splitting every cell from top left to bottom right.
    """
    def __init__(self, shape: Sequence[int], pbc=False):
        if len(shape) > 2: 
            raise NotImplementedError("Triangular lattices require at most 2 dimensions, {len(shape)} were given")
        self.shape = tuple(shape)
        # whether to assume periodic boundary conditions along individual axes
        if isinstance(pbc, bool):
            self.pbc = len(shape) * (pbc,)
        else:
            assert len(shape) == len(pbc)
            self.pbc = tuple(pbc)

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
        adj = np.zeros((self.nsites, self.nsites), dtype=int)
        idx = np.arange(self.nsites).reshape(self.shape)
        for d in range(self.ndim):
            for s in [-1, 1]:
                ids = np.roll(idx, s, axis=d)
                if self.pbc[d]:
                    for (i, j) in zip(idx.reshape(-1), ids.reshape(-1)):
                        adj[i, j] = 1
                else:
                    # single out axis `d`
                    seld = (math.prod(self.shape[:d]), self.shape[d], math.prod(self.shape[d+1:]))
                    idx_cut = idx.reshape(seld)
                    ids_cut = ids.reshape(seld)
                    if s == 1:
                        idx_cut = idx_cut[:, 1:, :]
                        ids_cut = ids_cut[:, 1:, :]
                    elif s == -1:
                        idx_cut = idx_cut[:, :-1, :]
                        ids_cut = ids_cut[:, :-1, :]
                    else:
                        assert False
                    for (i, j) in zip(idx_cut.reshape(-1), ids_cut.reshape(-1)):
                        adj[i, j] = 1
        # diagonal edge
        for d in range(self.ndim-1):
            for s in [-1, 1]:
                ids = np.roll(idx, s, axis=0)
                ids = np.roll(ids, s, axis=1)
                if self.pbc[d]:
                    for (i, j) in zip(idx.reshape(-1), ids.reshape(-1)):
                        adj[i, j] = 1
                else:
                    if self.pbc[d+1]:
                        seld = (math.prod(self.shape[:d]), self.shape[d], math.prod(self.shape[d+1:]))
                        idx_cut = idx.reshape(seld)
                        ids_cut = ids.reshape(seld)
                        if s == 1:
                            idx_cut = idx_cut[:, 1:, :]
                            ids_cut = ids_cut[:, 1:, :]
                        elif s == -1:
                            idx_cut = idx_cut[:, :-1, :]
                            ids_cut = ids_cut[:, :-1, :]
                        else:
                            assert False
                    else:
                    # single out axis `d` and `d+1`
                        seld = (math.prod(self.shape[:d]), self.shape[d], self.shape[d+1], math.prod(self.shape[d+2:]))
                        idx_cut = idx.reshape(seld)
                        ids_cut = ids.reshape(seld)
                        if s == 1:
                            idx_cut = idx_cut[:, 1:, 1:, :]
                            ids_cut = ids_cut[:, 1:, 1:, :]
                        elif s == -1:
                            idx_cut = idx_cut[:, :-1, :-1, :]
                            ids_cut = ids_cut[:, :-1, :-1, :]
                        else:
                            assert False
                    for (i, j) in zip(idx_cut.reshape(-1), ids_cut.reshape(-1)):
                        adj[i, j] = 1
        return adj

    def index_to_coord(self, i: int) -> tuple:
        """
        Map linear index to lattice coordinate.
        """
        return np.unravel_index(i, self.shape)

    def coord_to_index(self, c) -> int:
        """
        Map lattice coordinate to linear index.
        """
        return int(np.ravel_multi_index(c, self.shape))
