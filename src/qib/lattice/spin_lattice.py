import math
import numpy as np
from copy import copy
from typing import Sequence
from qib.lattice import AbstractLattice


class SpinLattice(AbstractLattice):
    """
    Lattice used for representing fermions with spin.
    Consists in 2 (or more) connected layers of the same lattice.
    """
    def __init__(self, lattice: AbstractLattice, nspin=2):
        if not isinstance(nspin, int) or nspin < 1:
            raise NotImplementedError(f"Spin dimension not supported.")
        self.shape = lattice.shape + (nspin,)
        self.original_lattice = lattice
        self.nspin = nspin

    @property
    def nsites(self) -> int:
        """
        Number of lattice sites.
        """
        return int(self.nspin*self.original_lattice.nsites)

    @property
    def ndim(self) -> int:
        """
        Number of spatial dimensions.
        """
        return len(self.shape)

    def adjacency_matrix(self):
        """
        Construct the adjacency matrix, indicating nearest neighbors.
        Block matrix with the original adjacency matrix on diagonal and identity on second diagonal.
        """
        original_adj = self.original_lattice.adjacency_matrix()
        connect = np.identity(self.original_lattice.nsites, dtype=int)
        return np.block([[connect]*i + [original_adj] + [connect]*(self.nspin-i-1) for i in range(self.nspin)])

    def index_to_coord(self, i: int) -> tuple:
        """
        Map linear index to lattice coordinate.
        It's trivial only in the integer lattice case, I have to rely on the original lattice equivalent function.
        """
        assert i < self.nsites
        return self.original_lattice.index_to_coord(i%self.original_lattice.nsites) + (i//self.original_lattice.nsites,)

    def coord_to_index(self, c) -> int:
        """
        Map lattice coordinate to linear index.
        It's trivial only in the integer lattice case, I have to rely on the original lattice equivalent function.
        """
        assert c[-1] < self.nspin
        return int(self.original_lattice.coord_to_index(c[:-1]) + self.original_lattice.nsites*c[-1])
