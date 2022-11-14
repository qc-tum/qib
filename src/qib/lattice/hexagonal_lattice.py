import math
from typing import Sequence
from qib.lattice import AbstractLattice
from qib.lattice.shifted_lattice_convention import ShiftedLatticeConvention
from qib.lattice.brick_lattice import BrickLattice


class HexagonalLattice(AbstractLattice):
    """
    Hexagonal lattice.
    The lattice has n full hexagons per row and m full hexagons per column.      
    """
    def __init__(self, shape: Sequence[int], pbc=False, convention: ShiftedLatticeConvention=ShiftedLatticeConvention.COLS_SHIFTED_UP):
        if len(shape) != 2: 
            raise NotImplementedError("Hexagonal lattices require 2 dimensions, {len(shape)} were given")
        self.shape = tuple(shape)
        self.convention = convention
        if pbc is True:
            # TODO: add pbc in adjacency matrix
            raise NotImplementedError("The hexagonal lattice doesn't hold periodic boundary conditions yet")
        self.pbc = pbc

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

    def equivalent_brick_lattice(self):
        """
        Returns the equivalent brick lattice (delete=True)
         _   _   _      _   _   _
        / \_/ \_/ \    | |_| |_| |
        \_/ \_/ \_/ == |_| |_| |_|
          \_/ \_/      . |_| |_| .
          
        """
        # Calculates shape, including 2 possible extra points
        return BrickLattice(shape=self.shape, pbc=self.pbc, delete=True, convention=self.convention)

    def adjacency_matrix(self):
        """
        Construct the adjacency matrix, indicating nearest neighbors.
        Adj matrix built from the equivalent brick lattice (delete=True).
        """
        # An equivalent brick graph is built.
        equiv_lattice = self.equivalent_brick_lattice()
        return equiv_lattice.adjacency_matrix()

    def index_to_coord(self, i: int) -> tuple:
        """
        Map linear index to the hexagonal lattice coordinates.
        """
        equiv_lattice = self.equivalent_brick_lattice()
        square_coord = equiv_lattice.index_to_coord(i)
        # TODO: let the user choose how the lattice scales
        length = 1.
        if self.convention == ShiftedLatticeConvention.COLS_SHIFTED_UP:
            if square_coord[0]%2==0:
                y = (0.5 + (square_coord[1]+1)//2 + 2.*(square_coord[1]//2))*length
            else:
                y = (square_coord[1]//2 + 2.*((square_coord[1]+1)//2))*length
            x = (square_coord[0]*math.sqrt(3)/2.)*length

        if self.convention == ShiftedLatticeConvention.ROWS_SHIFTED_LEFT:
            if square_coord[1]%2==0:
                x = (0.5 + (square_coord[0]+1)//2 + 2.*(square_coord[0]//2))*length
            else:
                x = (square_coord[0]//2 + 2.*((square_coord[0]+1)//2))*length
            y = (square_coord[1]*math.sqrt(3)/2.)*length

        return (x,y)

    def coord_to_index(self, c) -> int:
        """
        Map lattice coordinates to single index.
        """
        # TODO: let the user choose how the lattice scales and the tolerance
        length = 1.
        tol = 1e-8
        if self.convention == ShiftedLatticeConvention.COLS_SHIFTED_UP:
            x = c[0]*2./(length*math.sqrt(3.))
            int_x = round(x)
            if abs(x - int_x) > tol:
                 raise ValueError(f"Incompatible set of coordinates")
            y = c[1]/length
            if int_x%2 == 0:
                y -= 0.5
                int_y = round(y)
                if abs(y - int_y) > tol:
                    raise ValueError(f"Incompatible set of coordinates")
                int_y = 2*(int_y//3) + (int_y)%3
            else:
                int_y = round(y)
                if abs(y - int_y) > tol:
                    raise ValueError(f"Incompatible set of coordinates")
                int_y = 2*((int_y)//3) + (int_y%3)//2
                
            
        if self.convention == ShiftedLatticeConvention.ROWS_SHIFTED_LEFT:
            y = c[1]*2./(length*math.sqrt(3.))
            int_y = round(y)
            if abs(y - int_y) > tol:
                raise ValueError(f"Incompatible set of coordinates")
            x = c[0]/length
            if int_y%2 == 0:
                x -= 0.5
                int_x = round(x)
                if abs(x - int_x) > tol:
                    raise ValueError(f"Incompatible set of coordinates")
                int_x = 2*(int_x//3) + (int_x)%3
            else:
                int_x = round(x)
                if abs(x - int_x) > tol:
                    raise ValueError(f"Incompatible set of coordinates")
                int_x = 2*((int_x)//3) + (int_x%3)//2

        equiv_lattice = self.equivalent_brick_lattice()
        return equiv_lattice.coord_to_index((int_x,int_y))


