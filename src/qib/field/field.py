import enum
from qib.lattice import AbstractLattice


class ParticleType(enum.Enum):
    """
    Physical particle type.
    """
    QUBIT    = 1    # equivalent to a spin-1/2 particle
    QUDIT    = 2    # boson with a cut-off
    FERMION  = 3    # fermion
    MAJORANA = 4    # Majorana


class Field:
    """
    Particle "field" (i.e., a lattices of qubits, fermions, bosons...)
    """
    def __init__(self, ptype: ParticleType, latt: AbstractLattice):
        self.ptype = ptype
        self.lattice = latt

    @property
    def particle_type(self) -> ParticleType:
        """
        Particle type.
        """
        return self.ptype
