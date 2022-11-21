from qib.field import ParticleType, Field


class Particle:
    """
    Quantum particle, like a qubit, fermion, boson, ... as element of a field.
    The 'index' is the linear index into the underlying lattice.
    """
    def __init__(self, field: Field, index: int):
        self.field = field
        self.index = index

    @property
    def particle_type(self):
        """
        Particle type.
        """
        return self.field.particle_type

    def __eq__(self, other):
        """
        Logical equality test of two particles.
        """
        return self.field == other.field and self.index == other.index


class Qubit(Particle):
    """
    Qubit as a special quantum particle.
    """
    def __init__(self, field: Field=None, index: int=-1):
        # consistency check
        if field and field.particle_type != ParticleType.QUBIT:
            raise ValueError(f"expecting qubit particle type in underlying field, received {field.particle_type}")
        super().__init__(field, index)


class Boson(Particle):
    """
    Boson as a special quantum particle.
    """
    def __init__(self, field: Field=None, index: int=-1):
        # consistency check
        if field and field.particle_type != ParticleType.BOSON:
            raise ValueError(f"expecting boson particle type in underlying field, received {field.particle_type}")
        super().__init__(field, index)


class Fermion(Particle):
    """
    Fermion as a special quantum particle.
    """
    def __init__(self, field: Field=None, index: int=-1):
        # consistency check
        if field and field.particle_type != ParticleType.FERMION:
            raise ValueError(f"expecting fermion particle type in underlying field, received {field.particle_type}")
        super().__init__(field, index)
