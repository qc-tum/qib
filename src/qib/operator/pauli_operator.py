import numpy as np
from scipy import sparse
from typing import Sequence
from qib.operator import AbstractOperator
from qib.field import ParticleType, Field


class PauliString(AbstractOperator):
    """
    Pauli string (sequence of identity and Pauli matrices X, Y, Z),
    together with a global phase factor.

    Using check matrix representation (similar to Qiskit convention)
    by storing binary arrays `z` and `x`. The logical Pauli string is

    .. math:: (-i)^q \otimes_{k=0}^{n-1} (-i)^{z_k x_k} Z^{z_k} X^{x_k}

    Using convention that Pauli matrix at site 0 corresponds to slowest varying index.
    """
    def __init__(self, z, x, q):
        self.z = np.array(z, copy=False, dtype=int)
        self.x = np.array(x, copy=False, dtype=int)
        if self.z.ndim != 1 or self.x.ndim != 1:
            raise ValueError("'z' and 'x' parameters must be one-dimensional arrays")
        if self.z.shape != self.x.shape:
            raise ValueError("dimensions of 'z' and 'x' parameters must agree")
        if not set(self.z).issubset({ 0, 1 }):
            raise ValueError("only allowed entries in 'z' are 0 or 1")
        if not set(self.x).issubset({ 0, 1 }):
            raise ValueError("only allowed entries in 'x' are 0 or 1")
        self.q = int(q) % 4
        self.field = None

    @classmethod
    def identity(cls, nqubits: int):
        """
        Construct the Pauli string representation of the identity operation.
        """
        z = np.zeros(nqubits, dtype=int)
        x = np.zeros(nqubits, dtype=int)
        return cls(z, x, 0)

    @classmethod
    def from_string(cls, s: str):
        """
        Construct a Pauli string from a literal string representation, like "iYZXZ".
        """
        # remove whitespace
        s = s.replace(' ', '')
        # remove a potential leading '+' sign
        if s[0] == '+':
            s = s[1:]
        if s[0] == '-':
            if s[1] == 'i':
                q = 1
                s = s[2:]
            else:
                q = 2
                s = s[1:]
        elif s[0] == 'i':
            q = 3
            s = s[1:]
        else:
            q = 0
        return cls.from_single_paulis(len(s), *[list(reversed(x)) for x in enumerate(s)], q=q)

    @classmethod
    def from_single_paulis(cls, nqubits: int, *args, **kwargs):
        """
        Construct a Pauli string from a list of single Pauli matrices, like
        ('Y', 1), ('X', 4), ('Z', 5).
        """
        z = np.zeros(nqubits, dtype=int)
        x = np.zeros(nqubits, dtype=int)
        for arg in args:
            # unpack
            s, i = arg
            if i < 0 or i >= nqubits:
                raise ValueError(f"index {i} out of range, number of qubits is {nqubits}")
            # overwrite entries in case same index appears multiple times
            if s == 'I':
                z[i] = 0
                x[i] = 0
            elif s == 'X':
                z[i] = 0
                x[i] = 1
            elif s == 'Y':
                z[i] = 1
                x[i] = 1
            elif s == 'Z':
                z[i] = 1
                x[i] = 0
            else:
                raise ValueError(f"invalid Pauli matrix denominator, received {s}, expecting 'I', 'X', 'Y' or 'Z'")
        q = 0
        for key, value in kwargs.items():
            if key == "q":
                q = value
        return cls(z, x, q)

    def is_unitary(self):
        """
        Whether the operator is unitary.
        """
        return True

    def is_hermitian(self):
        """
        Whether the operator is Hermitian.
        """
        return self.q % 2 == 0

    @property
    def num_qubits(self):
        """
        Number of qubits, i.e., length of Pauli string, including identities.
        """
        return len(self.z)

    def get_pauli(self, i: int):
        """
        Get the Pauli matrix at index `i` as string 'I', 'X', 'Y' or 'Z'.
        """
        if self.z[i] == 0:
            if self.x[i] == 0:
                return 'I'
            else:
                return 'X'
        else:
            if self.x[i] == 0:
                return 'Z'
            else:
                return 'Y'

    def set_pauli(self, s: str, i: int):
        """
        Set an individual Pauli matrix at index `i`.
        """
        if s == 'I':
            self.z[i] = 0
            self.x[i] = 0
        elif s == 'X':
            self.z[i] = 0
            self.x[i] = 1
        elif s == 'Y':
            self.z[i] = 1
            self.x[i] = 1
        elif s == 'Z':
            self.z[i] = 1
            self.x[i] = 0
        else:
            raise ValueError(f"invalid Pauli matrix denominator, received {s}, expecting 'I', 'X', 'Y' or 'Z'")

    def refactor_phase(self):
        """
        "Refactor" the Pauli string by setting `q` to 0
        and returning the corresponding overall phase factor.
        """
        assert self.q < 4
        phase = [1., -1j, -1., 1j][self.q]
        self.q = 0
        return phase

    def refactor_sign(self):
        """
        "Refactor" the Pauli string by forcing `q` to be 0 or 1
        and returning the corresponding overall sign factor.
        """
        assert self.q < 4
        if self.q < 2:
            return 1
        else:
            self.q %= 2
            return -1

    def __matmul__(self, other):
        """
        Logical matrix multiplication of two Pauli strings.
        """
        z_prod = np.mod(self.z + other.z, 2)
        x_prod = np.mod(self.x + other.x, 2)
        q_prod = int(  np.dot(self.z, self.x)
                     + np.dot(other.z, other.x)
                     - np.dot(z_prod, x_prod)
                     + 2*np.dot(self.x, other.z))   # sign factor from flipping X <-> Z
        return PauliString(z_prod, x_prod, self.q + other.q + q_prod)

    def __eq__(self, other):
        """
        Test logical equality of Pauli strings.
        """
        return (np.array_equal(self.z, other.z)
            and np.array_equal(self.x, other.x)
            and self.q == other.q)

    def as_matrix(self):
        """
        Generate the sparse matrix representation of the Pauli string.
        """
        X = sparse.csr_matrix([[ 0.,  1.], [ 1.,  0.]])
        Z = sparse.csr_matrix([[ 1.,  0.], [ 0., -1.]])
        op = sparse.identity(1)
        for i in range(self.num_qubits):
            # using convention that Pauli matrix at site 0 corresponds to slowest varying index
            op = sparse.csr_matrix(sparse.kron(op, Z**self.z[i] @ X**self.x[i]))
            op.eliminate_zeros()
        # only use complex type when necessary
        phase = [1., -1j, -1., 1j][(self.q + np.dot(self.z, self.x)) % 4]
        return phase * op

    def set_field(self, field: Field):
        """
        Set a field which the Pauli string acts on
        (such that it can be used like a Hamiltonian).
        """
        if field.particle_type != ParticleType.QUBIT:
            raise ValueError(f"expecting a field with qubit particle type, but received {field.particle_type}")
        if field.lattice.nsites != self.num_qubits:
            raise ValueError(f"field lattice has {field.lattice.nsites} qubits, but Pauli string acts on {self.num_qubits} qubits")
        self.field = field
        # enable chaining
        return self

    def fields(self):
        """
        List of fields the Pauli string acts on.
        """
        if self.field is not None:
            return [self.field]
        else:
            return []

    def __str__(self):
        """
        Literal string representation of the Pauli string.
        """
        if self.q == 0:
            s = ""
        elif self.q == 1:
            s = "-i"
        elif self.q == 2:
            s = "-"
        elif self.q == 3:
            s = "i"
        else:
            assert False
        for i in range(self.num_qubits):
            s += self.get_pauli(i)
        return s


class WeightedPauliString(AbstractOperator):
    """
    Pauli string with a weight factor.
    """
    def __init__(self, paulis: PauliString, weight):
        self.paulis = paulis
        self.weight = weight

    def is_unitary(self):
        """
        Whether the operator is unitary.
        """
        return abs(self.weight) == 1

    def is_hermitian(self):
        """
        Whether the operator is Hermitian.
        """
        return ([1., -1j, -1., 1j][self.paulis.q] * self.weight).imag == 0

    @property
    def num_qubits(self):
        """
        Number of qubits, i.e., length of Pauli string, including identities.
        """
        return self.paulis.num_qubits

    def as_matrix(self):
        """
        Generate the sparse matrix representation of the weighted Pauli string.
        """
        return self.weight * self.paulis.as_matrix()

    def set_field(self, field: Field):
        """
        Set a field which the weighted Pauli string acts on
        (such that it can be used like a Hamiltonian).
        """
        self.paulis.set_field(field)
        # enable chaining
        return self

    def fields(self):
        """
        List of fields the weighted Pauli string acts on.
        """
        return self.paulis.fields()

    def __str__(self):
        """
        Literal string representation of the weighted Pauli string.
        """
        phase = [1., -1j, -1., 1j][self.paulis.q]
        s = str(phase * self.weight) + "*"
        for i in range(self.num_qubits):
            s += self.paulis.get_pauli(i)
        return s


class PauliOperator(AbstractOperator):
    """
    An operator consisting of Pauli strings.
    """
    def __init__(self, pstrings: Sequence[WeightedPauliString]=[]):
        # consistency check
        nqs = [ps.num_qubits for ps in pstrings]
        if len(set(nqs)) > 1:
            raise ValueError("all Pauli strings must have the same length")
        self.pstrings = list(pstrings)

    def add_pauli_string(self, ps: WeightedPauliString):
        """
        Add a weighted Pauli string. If the string already exists,
        only its weight is updated.
        """
        for pstring in self.pstrings:
            if pstring.paulis == ps.paulis:
                pstring.weight += ps.weight
                return
        self.pstrings.append(ps)

    def is_unitary(self):
        """
        Whether the operator is unitary.
        """
        raise NotImplementedError

    def is_hermitian(self):
        """
        Whether the operator is Hermitian.
        """
        return all(ps.is_hermitian() for ps in self.pstrings)

    @property
    def num_qubits(self):
        """
        Number of qubits, i.e., length of each Pauli string, including identities.
        """
        if self.pstrings:
            return self.pstrings[0].num_qubits
        else:
            # not specified
            return 0

    def as_matrix(self):
        """
        Generate the (sparse) matrix representation of the operator.
        """
        # note: dimensions are not specified if self.pstrings is empty
        op = 0
        for ps in self.pstrings:
            op += ps.as_matrix()
        return op

    def remove_zero_weight_strings(self, tol=0.0):
        """
        Remove the redundant Pauli strings with weight zero.
        """
        for i in range(len(self.pstrings)-1, -1, -1):
            # ensure that at least one weighted Pauli string remains
            # (even if it has zero weight) to retain dimension information
            if abs(self.pstrings[i].weight) <= tol and len(self.pstrings) > 1:
                self.pstrings.pop(i)

    def set_field(self, field: Field):
        """
        Set a field which the Pauli operator acts on
        (such that it can be used like a Hamiltonian).
        """
        for pstring in self.pstrings:
            pstring.set_field(field)
        # enable chaining
        return self

    def fields(self):
        """
        List of fields the Pauli operator acts on.
        """
        if self.pstrings:
            flist = self.pstrings[0].fields()
            # consistency check
            for pstring in self.pstrings:
                if not flist == pstring.fields():
                    raise RuntimeError("all weighted Pauli strings must act on the same field.")
            return flist
        else:
            return []

    def __str__(self):
        """
        String representation of the Pauli operator.
        """
        ps_list = [str(ps) for ps in self.pstrings]
        max_width = max([len(ps) for ps in ps_list])
        s = "Pauli operator consisting of weighted Pauli strings\n"
        for ps in ps_list:
            s += ps.rjust(max_width) + '\n'
        return s
