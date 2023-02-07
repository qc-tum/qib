import numpy as np
from typing import Sequence
from qib.field import ParticleType, Field
from qib.operator import AbstractOperator, FieldOperator, FieldOperatorTerm, IFOType, IFODesc


class BornOppenheimerHamiltonian(AbstractOperator):
    """
    Born-Oppenheimer Hamiltonian.
    Gets the one and two-body integrals on the SPIN-ORBITAL basis (same conventions as pyscf).
    Note that pyscf uses chemist's notation for MO integrals:
        V = sum_{i,j,k,l} v_{i,j,k,l} a+_i a_j a+_k a_l
    """
    def __init__(self, field: Field, h0: float = None, h1: Sequence[float] = None, h2: Sequence[float] = None):
        """
        Initialize the Hamiltonian through its SPIN-ORBITAL integrals.
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
        self.ao_mo = 'ao'

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

    def set_mo_integrals(self, coeff):
        """
        Change the spin-orbitals integral w.r.t. a coefficient matrix externally calculated (ex: Hartree Fock).
        """
        if self.ao_mo == 'mo':
            return
        
        spin_coeff = np.kron(coeff, np.identity(2))
        self.h1 = np.einsum('ji,jk,kl->il', spin_coeff.conj(), self.h1, spin_coeff)
        self.h2 = np.einsum('pqrs,pi,qj,rk,sl->ijkl', self.h2, spin_coeff, spin_coeff, spin_coeff, spin_coeff)
        self.ao_mo = 'mo'

    def reset_ao_integrals(self):
        """
        TODO: Reset atomic integrals.
        """
        return

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
                               IFODesc(self.field, IFOType.FERMI_ANNIHIL),
                               IFODesc(self.field, IFOType.FERMI_CREATE),
                               IFODesc(self.field, IFOType.FERMI_ANNIHIL)],
                               self.h2)
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