import numpy as np
from qib.field import ParticleType, Field
from qib.lattice import LayeredLattice
from qib.operator import AbstractOperator, FieldOperator, FieldOperatorTerm, IFOType, IFODesc


# TODO: possibility of non-uniform parameters?

class FermiHubbardHamiltonian(AbstractOperator):
    """
    Fermi-Hubbard Hamiltonian with
      - kinetic hopping coefficient `t`
      - potential interaction strength `u`
    on a lattice.
    """
    def __init__(self, field: Field, t: float, u: float, spin=True):
        # parameter checks
        if field.particle_type != ParticleType.FERMION:
            raise ValueError(f"expecting a field with fermionic particle type, but received {field.particle_type}")
        if not isinstance(t, float):
            raise ValueError(f"expecting a float for 't', received {type(t)}")
        if not isinstance(u, float):
            raise ValueError(f"expecting a float for 'u', received {type(u)}")
        if spin:
            if not isinstance(field.lattice, LayeredLattice):
                raise ValueError("expecting a layered lattice when 'spin' is True")
            if field.lattice.nlayers != 2:
                raise ValueError(f"layered lattice must have two layers (instead of {field.lattice.nlayers})")
        self.t = t
        self.u = u
        self.spin = spin
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

    def as_field_operator(self):
        """
        Represent the Hamiltonian as FieldOperator.
        """
        latt = self.field.lattice
        L = latt.nsites
        adj = latt.adjacency_matrix()
        if self.spin:
            assert L % 2 == 0
            kin_coeffs = -self.t * np.kron(np.identity(2), adj[:(L//2), :(L//2)])
            int_coeffs = np.zeros((L, L, L, L))
            for i in range(L//2):
                int_coeffs[i, i, i + L//2, i + L//2] = self.u
        else:
            kin_coeffs = -self.t * adj
            int_coeffs = np.zeros((L, L, L, L))
            for i in range(L):
                for j in range(i + 1, L):
                    if adj[i, j] != 0:
                        int_coeffs[i, i, j, j] = self.u
        # kinetic hopping term
        T = FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE),
                               IFODesc(self.field, IFOType.FERMI_ANNIHIL)],
                              kin_coeffs)
        # interaction term
        V = FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE),
                               IFODesc(self.field, IFOType.FERMI_ANNIHIL),
                               IFODesc(self.field, IFOType.FERMI_CREATE),
                               IFODesc(self.field, IFOType.FERMI_ANNIHIL)],
                              int_coeffs)
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
