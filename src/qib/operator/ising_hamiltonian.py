import enum
from qib.field import ParticleType, Field
from qib.operator import AbstractOperator, PauliString, WeightedPauliString, PauliOperator


class IsingConvention(enum.Enum):
    """
    Convention for the Ising Hamiltonian.
    """
    ISING_ZZ = 1    # "J Z Z + h Z + g X" convention
    ISING_XX = 2    # "J X X + h X + g Z" convention


class IsingHamiltonian(AbstractOperator):
    """
    Ising Hamiltonian "J Z Z + h Z + g X" or "J X X + h X + g Z" with
      - interaction parameter `J`
      - longitudinal field strength `h`
      - transverse field strength `g`
    on a lattice.
    """
    def __init__(self, field: Field, J: float, h: float, g: float,
                 convention: IsingConvention=IsingConvention.ISING_ZZ):
        # parameter checks
        if field.particle_type != ParticleType.QUBIT:
            raise ValueError(f"expecting a field with qubit particle type, but received {field.particle_type}")
        if not isinstance(J, (int, float)):
            raise ValueError(f"expecting real numeric value for 'J', received {J}")
        if not isinstance(h, (int, float)):
            raise ValueError(f"expecting real numeric value for 'h', received {h}")
        if not isinstance(g, (int, float)):
            raise ValueError(f"expecting real numeric value for 'g', received {g}")
        if not isinstance(convention, IsingConvention):
            raise ValueError(f"'convention' must be of type 'IsingConvention', received {convention}")
        self.field = field
        self.J = J
        self.h = h
        self.g = g
        self.convention = convention

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

    def as_pauli_operator(self):
        """
        Represent the Hamiltonian as Pauli operator.
        """
        latt = self.field.lattice
        L = latt.nsites
        adj = latt.adjacency_matrix()
        assert adj.shape == (L, L)
        op = PauliOperator()
        if self.convention == IsingConvention.ISING_ZZ:
            A = 'Z'
            B = 'X'
        else:
            A = 'X'
            B = 'Z'
        for i in range(L):
            for j in range(i + 1, L):
                if adj[i, j] == 0:
                    continue
                # interaction term
                # site 0 corresponds to fastest varying index
                op.add_pauli_string(WeightedPauliString(PauliString.from_single_paulis(L, (A, i), (A, j)), self.J))
            # longitudinal field term
            op.add_pauli_string(WeightedPauliString(PauliString.from_single_paulis(L, (A, i)), self.h))
            # transverse field term
            op.add_pauli_string(WeightedPauliString(PauliString.from_single_paulis(L, (B, i)), self.g))
        return op

    def as_matrix(self):
        """
        Generate the (sparse) matrix representation of the Hamiltonian.
        """
        op = self.as_pauli_operator()
        return op.as_matrix()

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
