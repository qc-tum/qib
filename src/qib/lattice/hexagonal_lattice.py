import math
import enum
import numpy as np
from typing import Sequence
from qib.lattice import AbstractLattice

class HexagonalLatticeConvention(enum.Enum):
    """
    Convention for the Hexagonal Lattice.
    Example with 1 row and 5 columns for COLS_SHIFTED_UP convention:
     _   _   _
    / \_/ \_/ \
    \_/ \_/ \_/
      \_/ \_/

    """
    COLS_SHIFTED_UP = 1       # The even numbered columns are shifted up relative to odd numbered columns.
    ROWS_SHIFTED_LEFT = 2     # The even numbered rows are shifted left relative to odd numbered rows.



class HexagonalLattice(AbstractLattice):
    """
    Hexagonal lattice.
    The lattice has n full hexagons per row and m full hexagons per column.      
    """
    def __init__(self, shape: Sequence[int], pbc=False, convention: HexagonalLatticeConvention=HexagonalLatticeConvention.COLS_SHIFTED_UP):
        if len(shape) != 2: 
            raise NotImplementedError("Hexagonal lattices require 2 dimensions, {len(shape)} were given")
        self.shape = tuple(shape)
        self.convention = convention
        self.shape_square = self._shape_square
        self.nsites_square = self._nsites_square
        if pbc is True:
            # TODO: add pbc in adjacency matrix
            raise NotImplementedError("The hexagonal lattice doesn't hold periodic boundary conditions yet")

    @property
    def nsites(self) -> int:
        """
        Number of lattice sites.
        """
        return 2*self.shape[0]*self.shape[1] +2*(self.shape[0]+self.shape[1])

    @property
    def ndim(self) -> int:
        """
        Number of spatial dimensions.
        """
        return len(self.shape)

    @property
    def _shape_square(self) -> tuple:
        """
        Shape of the equivalent square lattice.
        Includes the 2 extra points.
        """
        if self.convention == HexagonalLatticeConvention.COLS_SHIFTED_UP:
            if self.shape[1]>1:
                nrows_square = 2*self.shape[0]+2 
            else:
                nrows_square = 2*self.shape[0]+1 
            ncols_square = self.shape[1]+1
        if self.convention == HexagonalLatticeConvention.ROWS_SHIFTED_LEFT:
            if self.shape[0]>1:
                ncols_square = 2*self.shape[1]+2
            else:
                ncols_square = 2*self.shape[1]+1
            nrows_square = self.shape[0]+1
        return (nrows_square,ncols_square)

    @property
    def _nsites_square(self) -> int:
        """
        Number of lattice sites in the equivalent square lattice.
        Includes the 2 extra points.
        """
        return self.shape_square[0]*self.shape_square[1]

    def adjacency_matrix(self, delete=True):
        """
        Construct the adjacency matrix, indicating nearest neighbors.
        Hexagonal lattice equivalent to a rectangular one:
         _   _   _      _   _   _
        / \_/ \_/ \    | |_| |_| |
        \_/ \_/ \_/ == |_| |_| |_|
          \_/ \_/      . |_| |_| .
          
        """
        # An equivalent square graph is built.
        # For the COLS_SHIFTED_UP case (opposite for ROWS_SHIFTED_LEFT):
        # nrows is the number of points in the hexagonal lattice 
        # nrows is the number of points in the hexagonal lattice 
        if self.convention == HexagonalLatticeConvention.COLS_SHIFTED_UP:
            d_square = 0
            parity_shift_condition = (self.shape[1]%2 == 1)
            
        if self.convention == HexagonalLatticeConvention.ROWS_SHIFTED_LEFT:
            d_square = 1
            parity_shift_condition = (self.shape[0]>1)
        
        adj = np.zeros((self.nsites_square, self.nsites_square), dtype=int)
        idx = np.arange(self.nsites_square).reshape(self.shape_square)
        # the y axis for COLS_SHIFTED_UP and x axis for ROWS_SHIFTED_LEFT are treated like the square graph case.  
        # the other axis only has half of the connections.
        for d in range(self.ndim):
            for s in [-1, 1]:
                ids = np.roll(idx, s, axis=d)
                # single out axis `d`
                seld = (math.prod(self.shape_square[:d]), self.shape_square[d], math.prod(self.shape_square[d+1:]))
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
                if d == d_square:
                    for (i, j) in zip(idx_cut.reshape(-1), ids_cut.reshape(-1)):
                        adj[i, j] = 1
                else:
                    for (i, j) in zip(idx_cut.reshape(-1), ids_cut.reshape(-1)):
                        if parity_shift_condition:
                            if (s == -1 and (i+i//self.shape_square[1])%2 == 0) or (s == 1 and (i+i//self.shape_square[1])%2 == 1):
                                adj[i, j] = 1    
                        else:
                            if (s == -1 and i%2 == 0) or (s == 1 and i%2 == 1):
                                adj[i, j] = 1  
        if delete:
            adj = self._delete_extra_points(adj, self.shape_square)
        return adj

    def _delete_extra_points(self, adj, shape):
        # two extra points, they need to be eliminated.
        if self.convention == HexagonalLatticeConvention.COLS_SHIFTED_UP and self.shape[1]>1:
            adj = np.delete(adj, (shape[0]-1)*shape[1], 0)
            adj = np.delete(adj, (shape[0]-1)*shape[1], 1)
            if shape[1]%2 == 0:
                adj = np.delete(adj, adj.shape[0]-1, 0)
                adj = np.delete(adj, adj.shape[1]-1, 1)
            else:
                adj = np.delete(adj, shape[1]-1, 0)
                adj = np.delete(adj, shape[1]-1, 1)
        if self.convention == HexagonalLatticeConvention.ROWS_SHIFTED_LEFT and self.shape[0]>1:
            adj = np.delete(adj, shape[1]-1, 0)
            adj = np.delete(adj, shape[1]-1, 1)
            if shape[0]%2 == 1:
                adj = np.delete(adj, (shape[0]-1)*shape[1]-1, 0)
                adj = np.delete(adj, (shape[0]-1)*shape[1]-1, 1)
            else:
                adj = np.delete(adj, adj.shape[0]-1, 0)
                adj = np.delete(adj, adj.shape[1]-1, 1)
                
        return adj

    def index_to_coord(self, i: int, delete=True) -> tuple:
        """
        Map linear index to the equivalent square lattice coordinate.
        If delete=True the two extra points of the equivalent square lattice are not counted in.
        """
        shift = 0
        if delete:
            assert i < self.nsites
            if self.convention == HexagonalLatticeConvention.COLS_SHIFTED_UP and self.shape[1] > 1:
                if i >= self.shape_square[1]-1 and self.shape[1]%2 == 0:
                    shift += 1
                if i >= (self.shape_square[0]-1)*self.shape_square[1]-shift:
                    shift += 1
            if self.convention == HexagonalLatticeConvention.ROWS_SHIFTED_LEFT and self.shape[0] > 1:
                if i >= self.shape_square[1]-1:
                    shift += 1
                if i >= (self.shape_square[0]-1)*self.shape_square[1]-shift and  self.shape[0]%2 == 0:
                    shift += 1
                
        return np.unravel_index((i+shift), self.shape_square)

    def coord_to_index(self, c, delete=True) -> int:
        """
        Map lattice coordinate to the equivalent square lattice coordinate.
        If delete=True the two extra points of the equivalent square lattice are not counted in.
        """
        shift = 0
        if delete:
            if self.convention == HexagonalLatticeConvention.COLS_SHIFTED_UP and self.shape[1] > 1:
                # even and odd columns specific cases
                if self.shape[1]%2 == 0:
                    if c[0] == 0 and c[1] == self.shape_square[1]-1:
                        return None
                    elif c[0] > 0:
                        shift += 1
                else:
                    if c[0] == self.shape_square[0]-1 and c[1] == self.shape_square[1]-1:
                        return None
                # common shift for even and odd cases
                if c[0] == self.shape_square[0]-1:
                    if c[1] == 0:
                        return None
                    else:
                        shift += 1
            if self.convention == HexagonalLatticeConvention.ROWS_SHIFTED_LEFT and self.shape[0] > 1:
                # even and odd columns specific cases
                if self.shape[0]%2 == 0:
                    if c[0] == self.shape_square[0]-1:
                        if c[1] == 0:
                            return None
                        else:
                            shift += 1
                else:
                    if c[0] == self.shape_square[0]-1 and c[1] == self.shape_square[1]-1:
                        return None
                # common shift for even and odd cases
                if c[0] == 0:
                    if c[1] == self.shape_square[1]-1:
                        return None
                else:
                    shift += 1
        return int(np.ravel_multi_index(c, self.shape_square)) - shift
            
