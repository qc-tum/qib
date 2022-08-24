import math
import numpy as np
from typing import Sequence
from qib.lattice import AbstractLattice


class IntegerLattice(AbstractLattice):
    """
    n-dimensional integer lattice.
    """
    def __init__(self, shape: Sequence[int], pbc=False):
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
        return adj

    def adjacency_matrix_axis_shift(self, d: int, s: int):
        """
        Construct the adjacency matrix along axis `d` and shift `s`.
        """
        if not s in [-1, 1]:
            raise ValueError(f"invalid shift s = {s}, must be 1 or -1")
        adj = np.zeros((self.nsites, self.nsites), dtype=int)
        idx = np.arange(self.nsites).reshape(self.shape)
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
