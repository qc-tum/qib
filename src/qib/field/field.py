import enum
from qib.lattice import AbstractLattice


class ParticleType(enum.Enum):
    """
    Physical particle type.
    """
    QUBIT    = 1    # equivalent to a spin-1/2 particle
    BOSON    = 2    # boson (requires a cut-off for simulation)
    FERMION  = 3    # fermion
    MAJORANA = 4    # Majorana


class Field:
    """
    Particle "field" (i.e., a lattices of qubits, fermions, bosons...)
    """
    def __init__(self, ptype: ParticleType, latt: AbstractLattice, maxocc=0):
        self.ptype = ptype
        self.lattice = latt
        if ptype == ParticleType.BOSON:
            # maximum occupancy used in simulations
            if maxocc <= 0:
                raise ValueError("maximum occupancy must be positive for bosons")
            self.maxocc = maxocc

    @property
    def particle_type(self) -> ParticleType:
        """
        Particle type.
        """
        return self.ptype

    @property
    def local_dim(self):
        """
        Local dimension at a lattice site.
        """
        if self.ptype == ParticleType.BOSON:
            return self.maxocc + 1
        else:
            return 2

    def dof(self):
        """
        Overall degrees of freedom (quantum Hilbert space dimension),
        using maximum occupancy for bosons.
        """
        return self.local_dim**self.lattice.nsites
