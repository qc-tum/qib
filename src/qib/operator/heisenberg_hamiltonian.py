from typing import Sequence
from qib.field import ParticleType, Field
from qib.operator import AbstractOperator, PauliString, WeightedPauliString, PauliOperator


class HeisenbergHamiltonian(AbstractOperator):
    """
    Heisenberg Hamiltonian "J_1 X X + J_2 Y Y + J_3 Z Z + h_1 X + h_2 Y + h_3 Z"
      - interaction parameters `J`
      - longitudinal field strengths `h`
    on a lattice.
    """
    def __init__(self, field: Field, J: Sequence[float], h: Sequence[float]):
        # parameter checks
        if field.particle_type != ParticleType.QUBIT:
            raise ValueError(f"expecting a field with qubit particle type, but received {field.particle_type}")
        if not (len(J) == 3 and len(h) == 3):
            raise ValueError(f"expecting 3 components of 'J' and 'h', received {len(J)} and {len(h)}")
        for i in range(3):
            if not (isinstance(J[i], int) or isinstance(J[i], float)):
                raise ValueError(f"expecting real numeric value for J[{i}], received {J[i]}")
            if not (isinstance(h[i], int) or isinstance(h[i], float)):
                raise ValueError(f"expecting real numeric value for h[{i}], received {h[i]}")

        self.field = field
        self.J = tuple(J)
        self.h = tuple(h)

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

        for k, gate in enumerate(['X', 'Y', 'Z']):
            for i in range(L):
                for j in range(i + 1, L):
                    if adj[i, j] == 0:
                        continue
                    # interaction term
                    # site 0 corresponds to fastest varying index
                    op.add_pauli_string(WeightedPauliString(PauliString.from_single_paulis(L, (gate, i), (gate, j)), self.J[k]))
                # field term
                op.add_pauli_string(WeightedPauliString(PauliString.from_single_paulis(L, (gate, i)), self.h[k]))
        return op

    def as_matrix(self):
        """
        Generate the (sparse) matrix representation of the Hamiltonian.
        """
        op = self.as_pauli_operator()
        return op.as_matrix()

    def fields(self):
        """
        List of fields the Hamiltonian acts on.
        """
        return [self.field]
