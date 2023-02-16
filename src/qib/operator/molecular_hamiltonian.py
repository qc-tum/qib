import numpy as np
from typing import Sequence
from qib.field import ParticleType, Field
from qib.operator import AbstractOperator, FieldOperator, FieldOperatorTerm, IFOType, IFODesc


class MolecularHamiltonian(AbstractOperator):
    """
    Molecular Hamiltonian.
    Constructs a molecular Hamiltonian starting from the one andtwo body integrals.
    We use physcs' notation for MO integrals:
    :math:`V = 0.5 \sum_{i,j,k,\ell} v_{i,j,k,\ell} a^{\dagger}_i a^{\dagger}_j a_\ell a_k`
    """
    def __init__(self, field: Field, h0: float = None, h1: Sequence[float] = None, h2: Sequence[float] = None):
        """
        Initialize the Hamiltonian through its spin-orbital integrals.
        """
        self.n_spinor = len(h1)
        self.h0 = h0
        self.h1 = np.array(h1)
        self.h2 = np.array(h2)
        if h1.shape != (self.n_spinor, self.n_spinor):
            raise RuntimeError(f"h1 must have size ({self.n_spinor},{self.n_spinor}), while {h1.shape} was given")
        if h2.shape != (self.n_spinor, self.n_spinor, self.n_spinor, self.n_spinor):
            raise RuntimeError(f"hs must have size ({self.n_spinor},{self.n_spinor},{self.n_spinor},{self.n_spinor}), while {h2.shape} was given")
        if field.particle_type != ParticleType.FERMION:
            raise ValueError(f"expecting a field with fermionic particle type, but received {field.particle_type}")
        if field.lattice.nsites != self.n_spinor:
            raise ValueError(f"the lattice must have {2*self.n_spinor} sites, while {field.lattice.nsites} were given.")
        self.field = field

    def is_unitary(self):
        """
        Whether the Hamiltonian is unitary.
        """
        # unitary only in some non-typical cases,
        # so returning False here for simplicity
        return False

    def is_hermitian(self):
        """
        Whether the Hamiltonian is Hermitian.
        """
        return True

    def set_h0(self, h0: float):
        """
        Set the constant term of the Hamiltonian.
        """
        self.h0 = h0

    def set_h1(self, h1: Sequence[float]):
        """
        Set the one-body term of the Hamiltonian.
        """
        if h1.shape != (self.n_spinor, self.n_spinor):
            raise RuntimeError(f"h1 must have size ({self.n_spinor},{self.n_spinor}), while {h1.shape} was given")
        self.h1 = np.array(h1)

    def set_h2(self, h2: Sequence[float]):
        """
        Set the two-body term of the Hamiltonian.
        """
        if h2.shape != (self.n_spinor, self.n_spinor, self.n_spinor, self.n_spinor):
            raise RuntimeError(f"hs must have size ({self.n_spinor},{self.n_spinor},{self.n_spinor},{self.n_spinor}), while {h2.shape} was given")
        self.h2 = np.array(h2)

    def as_field_operator(self):
        """
        Represent the Hamiltonian as FieldOperator.
        TODO: take into account also h0
        """
        # kinetic hopping term
        T = FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE),
                               IFODesc(self.field, IFOType.FERMI_ANNIHIL)],
                               self.h1)
        # interaction term
        V = FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE),
                               IFODesc(self.field, IFOType.FERMI_CREATE),
                               IFODesc(self.field, IFOType.FERMI_ANNIHIL),
                               IFODesc(self.field, IFOType.FERMI_ANNIHIL)],
                               self.h2*0.5)
        return FieldOperator([T, V])

    def as_matrix(self):
        """
        Generate the (sparse) matrix representation of the Hamiltonian.
        """
        return self.as_field_operator().as_matrix()

    @property
    def nsites(self) -> int:
        """
        Number of underlying lattice sites.
        """
        return self.field.lattice.nsites

    def fields(self):
        """
        List of fields the Hamiltonian acts on.
        """
        return [self.field]
