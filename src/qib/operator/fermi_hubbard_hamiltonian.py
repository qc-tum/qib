import enum
import numpy as np
from typing import Sequence, Union
from qib.field import ParticleType, Field
from qib.lattice import SpinLattice
from qib.operator import AbstractOperator, FieldOperator, FieldOperatorTerm, IFOType, IFODesc

# TODO: possibility of non-uniform parameters?

class FermiHubbardHamiltonian(AbstractOperator):
    """
    Fermi - Hubbard Hamiltonian:
      - interaction matrix `t`
      - potential term `u`
    on a lattice.
    """
    def __init__(self, field: Field, t: float, u: float, spin=False):
        # parameter checks
        if field.particle_type != ParticleType.FERMION:
            raise ValueError(f"expecting a field with fermionic particle type, but received {field.particle_type}")
        if not isinstance(t, float):
            raise ValueError(f"expecting a float for 't', received {type(t)}")

        self.t = t
        self.u = u
        if spin and not isinstance(field.lattice, SpinLattice):
            raise ValueError(f"expecting a spin lattice when 'spin' is True")
        self.spin = spin
        self.field = field
        

    def is_unitary(self):
        """
        Whether the Hamiltonian is unitary.
        """
        H = self.as_matrix().toarray()
        return np.allclose(H@H, np.identity(self.field.lattice.nsites))

    def is_hermitian(self):
        """
        Whether the Hamiltonian is Hermitian.
        """
        return True

    def as_field_operator(self):
        """
        Represent the Hamiltonian as FieldOperator.
        """
        latt = self.field.lattice
        L = latt.nsites
        adj = latt.adjacency_matrix()
        assert adj.shape == (L,L)
        if self.spin:
            t = self.t*np.kron(np.identity(L), adj[:L, :L])
            u = np.zeros((L,L,L,L))
            for i in range(L):
                u[i,i,i,i] = self.u
            # kinetic term (only nearest neighbours)
            kin_ops = FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE),IFODesc(self.field, IFOType.FERMI_ANNIHIL)],t)
            # potential term (only on-site)
            pot_ops = FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE), IFODesc(self.field, IFOType.FERMI_ANNIHIL), IFODesc(self.field, IFOType.FERMI_CREATE), IFODesc(self.field, IFOType.FERMI_ANNIHIL)],u)
            
        else:
            t = self.t*adj
            u = self.u*np.identity(len(adj))
            # kinetic term (only nearest neighbours)
            kin_ops = FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE),IFODesc(self.field, IFOType.FERMI_ANNIHIL)],t)
            # potential term (only on-site)
            pot_ops = FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE), IFODesc(self.field, IFOType.FERMI_ANNIHIL)],u)
        return FieldOperator([kin_ops, pot_ops])

    def as_matrix(self):
        """
        Generate the (sparse) matrix representation of the Hamiltonian.
        """
        op = self.as_field_operator()
        return op.as_matrix()

    def fields(self):
        """
        List of fields the Hamiltonian acts on.
        """
        return [self.field]
