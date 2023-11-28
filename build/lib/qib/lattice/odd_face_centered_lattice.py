import math
from typing import Sequence
import numpy as np
from qib.lattice import AbstractLattice


class OddFaceCenteredLattice(AbstractLattice):
    """
    Modified square lattice in two dimensions, where every "odd" face
    has an additional vertex at the center. Used for "compact encoding".
    """
    def __init__(self, shape: Sequence[int], pbc=False):
        if len(shape) != 2:
            raise ValueError("currently only two-dimensional lattices supported")
        self.shape = tuple(shape)
        # whether to assume periodic boundary conditions along individual axes
        if isinstance(pbc, bool):
            self.pbc = len(shape) * (pbc,)
        else:
            assert len(shape) == len(pbc)
            self.pbc = tuple(pbc)
        for i, n in enumerate(self.shape):
            if n % 2 == 1 and self.pbc[i]:
                raise ValueError("each axis with periodic boundary conditions must have even dimension")

    @property
    def nsites(self) -> int:
        """
        Number of lattice sites.
        """
        return math.prod(self.shape) + ((self.shape[0] - 1)*(self.shape[1] - 1) + 1) // 2

    @property
    def ndim(self) -> int:
        """
        Number of spatial dimensions.
        """
        return len(self.shape)

    def adjacency_matrix(self):
        """
        Construct the adjacency matrix, indicating nearest neighbors.
        Each site at an odd face center is considered to be a neighbor
        of the four corners of the face.
        """
        adj = np.zeros((self.nsites, self.nsites), dtype=int)
        nverts = math.prod(self.shape)
        # conventional adjacency of an integer lattice
        idx = np.arange(nverts).reshape(self.shape)
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
        # adjacency between face centers and corners
        i = nverts  # counter
        for x in range(self.shape[0] - 1):
            for y in range(self.shape[1] - 1):
                if (x + y) % 2 == 1:
                    continue
                # enumerate the four corners of the face
                j = x*self.shape[1] + y
                adj[i, j] = 1
                adj[j, i] = 1
                j = x*self.shape[1] + (y + 1)
                adj[i, j] = 1
                adj[j, i] = 1
                j = (x + 1)*self.shape[1] + y
                adj[i, j] = 1
                adj[j, i] = 1
                j = (x + 1)*self.shape[1] + (y + 1)
                adj[i, j] = 1
                adj[j, i] = 1
                i += 1
        return adj

    def index_to_coord(self, i: int) -> tuple:
        """
        Map linear index to lattice coordinate.
        """
        nverts = math.prod(self.shape)
        if i < nverts:
            return np.unravel_index(i, self.shape)
        assert len(self.shape) == 2
        i -= nverts
        x = 2 * i // (self.shape[1] - 1)
        y = 2 * (i - (x*(self.shape[1] - 1) + 1) // 2) + (x % 2)
        return (x + 0.5, y + 0.5)

    def coord_to_index(self, c) -> int:
        """
        Map lattice coordinate to linear index.
        """
        idx = np.array(c)
        if np.issubdtype(idx.dtype, np.integer):
            return int(np.ravel_multi_index(idx, self.shape))
        if idx.shape != (2,):
            raise ValueError("multi-index must be a sequence of length 2")
        idx = np.round(idx - 0.5)
        x, y = int(idx[0]), int(idx[1])
        if (x + y) % 2 == 1:
            raise ValueError(f"multi-index {c} is not a valid odd face index")
        if x < 0 or y < 0 or x >= self.shape[0] - 1 or y >= self.shape[1] - 1:
            # face outside of the rectangular region
            raise ValueError(f"coordinates {c} outside of the rectangular region")
        # take offset by primary vertex qubits into account
        return math.prod(self.shape) + (x*(self.shape[1] - 1) + 1) // 2 + (y // 2)

    def edge_to_odd_face_index(self, i, j):
        """
        Find the adjacent "odd" face of edge (i, j),
        returning the lattice site index of the face,
        or -1 in case the edge is not adjacent to an odd face.
        """
        # unpack vertex coordinates
        ix, iy = i
        jx, jy = j
        # ensure that i, j are nearest neighbors
        if not((ix == jx and abs(iy - jy) == 1) or (iy == jy and abs(ix - jx) == 1)):
            raise ValueError(f"{i} - {j} is not a nearest neighbor edge")
        x = min(ix, jx)
        y = min(iy, jy)
        if x < 0 or y < 0 or x >= self.shape[0] or y >= self.shape[1]:
            raise ValueError(f"edge {i} - {j} is outside of the rectangular region")
        if (x + y) % 2 == 1:
            if ix == jx:
                x -= 1
            else:
                y -= 1
        if x < 0 or y < 0 or x >= self.shape[0] - 1 or y >= self.shape[1] - 1:
            # face outside of the rectangular region
            return -1
        # take offset by primary vertex qubits into account
        return math.prod(self.shape) + (x*(self.shape[1] - 1) + 1) // 2 + (y // 2)
