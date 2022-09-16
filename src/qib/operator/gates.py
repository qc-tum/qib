import abc
import numpy as np
from scipy.linalg import expm, block_diag
from scipy.sparse import csr_matrix
from typing import Sequence
from qib.field import Field, Particle, Qubit
from qib.operator import AbstractOperator, FieldOperator


class Gate(AbstractOperator):
    """
    Parent class for quantum gates.

    A quantum gate represents the corresponding unitary operation,
    and can store the qubits (or in general quantum particles) it acts on.
    """

    def is_unitary(self):
        """
        A quantum gate is unitary by definition.
        """
        return True

    @property
    @abc.abstractmethod
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        pass

    @abc.abstractmethod
    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        pass

    @abc.abstractmethod
    def fields(self):
        """
        Return the list fields hosting the quantum particles which the gate acts on.
        """
        pass

    @abc.abstractmethod
    def _circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        pass


class PauliXGate(Gate):
    """
    Pauli-X gate.
    """
    def __init__(self, qubit: Qubit=None):
        self.qubit = qubit

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return True

    def as_matrix(self):
        """
        Generate the matrix representation of the Pauli-X gate.
        """
        return np.array([[ 0.,  1.], [ 1.,  0.]])

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return 1

    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        if self.qubit:
            return [self.qubit]
        else:
            return []

    def fields(self):
        """
        Return the list fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        else:
            return []

    def on(self, qubit: Qubit):
        """
        Act on the specified qubit.
        """
        self.qubit = qubit
        # enable chaining
        return self

    def _circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.qubit:
            raise RuntimeError("unspecified target qubit")
        iwire = _map_particle_to_wire(fields, self.qubit)
        if iwire < 0:
            raise RuntimeError("qubit not found among fields")
        nwires = sum([f.lattice.nsites for f in fields])
        return _distribute_to_wires(nwires, [iwire], csr_matrix(self.as_matrix()))


class PauliYGate(Gate):
    """
    Pauli-Y gate.
    """
    def __init__(self, qubit: Qubit=None):
        self.qubit = qubit

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return True

    def as_matrix(self):
        """
        Generate the matrix representation of the Pauli-Y gate.
        """
        return np.array([[ 0., -1j], [ 1j,  0.]])

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return 1

    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        if self.qubit:
            return [self.qubit]
        else:
            return []

    def fields(self):
        """
        Return the list fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        else:
            return []

    def on(self, qubit: Qubit):
        """
        Act on the specified qubit.
        """
        self.qubit = qubit
        # enable chaining
        return self

    def _circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.qubit:
            raise RuntimeError("unspecified target qubit")
        iwire = _map_particle_to_wire(fields, self.qubit)
        if iwire < 0:
            raise RuntimeError("qubit not found among fields")
        nwires = sum([f.lattice.nsites for f in fields])
        return _distribute_to_wires(nwires, [iwire], csr_matrix(self.as_matrix()))


class PauliZGate(Gate):
    """
    Pauli-Z gate.
    """
    def __init__(self, qubit: Qubit=None):
        self.qubit = qubit

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return True

    def as_matrix(self):
        """
        Generate the matrix representation of the Pauli-Z gate.
        """
        return np.array([[ 1.,  0.], [ 0., -1.]])

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return 1

    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        if self.qubit:
            return [self.qubit]
        else:
            return []

    def fields(self):
        """
        Return the list fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        else:
            return []

    def on(self, qubit: Qubit):
        """
        Act on the specified qubit.
        """
        self.qubit = qubit
        # enable chaining
        return self

    def _circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.qubit:
            raise RuntimeError("unspecified target qubit")
        iwire = _map_particle_to_wire(fields, self.qubit)
        if iwire < 0:
            raise RuntimeError("qubit not found among fields")
        nwires = sum([f.lattice.nsites for f in fields])
        return _distribute_to_wires(nwires, [iwire], csr_matrix(self.as_matrix()))


class HadamardGate(Gate):
    """
    Hadamard gate.
    """
    def __init__(self, qubit: Qubit=None):
        self.qubit = qubit

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return True

    def as_matrix(self):
        """
        Generate the matrix representation of the Hadamard gate.
        """
        return np.array([[ 1.,  1.], [ 1., -1.]]) / np.sqrt(2)

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return 1

    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        if self.qubit:
            return [self.qubit]
        else:
            return []

    def fields(self):
        """
        Return the list fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        else:
            return []

    def on(self, qubit: Qubit):
        """
        Act on the specified qubit.
        """
        self.qubit = qubit
        # enable chaining
        return self

    def _circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.qubit:
            raise RuntimeError("unspecified target qubit")
        iwire = _map_particle_to_wire(fields, self.qubit)
        if iwire < 0:
            raise RuntimeError("qubit not found among fields")
        nwires = sum([f.lattice.nsites for f in fields])
        return _distribute_to_wires(nwires, [iwire], csr_matrix(self.as_matrix()))


class ControlledGate(Gate):
    """
    A controlled quantum gate with an arbitrary number of control qubits.

    Use the reference to the target gate to set the qubits (or particles) it acts on.
    """
    def __init__(self, tgate: Gate, ncontrols: int):
        self.tgate = tgate
        self.ncontrols = ncontrols
        self.control_qubits = []

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return self.tgate.is_hermitian()

    def as_matrix(self):
        """
        Generate the matrix representation of the controlled gate.
        """
        tgmat = self.tgate.as_matrix()
        # target gate corresponds to faster varying indices
        return block_diag(np.identity((2**self.ncontrols - 1) * tgmat.shape[0]), tgmat)

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return self.tgate.num_wires + self.ncontrols

    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        return self.tgate.particles() + self.control_qubits

    def fields(self):
        """
        Return the list fields hosting the quantum particles which the gate acts on.
        """
        return list(set(self.tgate.fields() + [q.field for q in self.control_qubits]))

    def target_gate(self):
        """
        Get the target gate.
        """
        return self.tgate

    @property
    def num_controls(self):
        """
        The number of control qubits.
        """
        return self.ncontrols

    def set_control(self, *args):
        """
        Set the control qubits.
        """
        if len(args) == 1 and isinstance(args[0], Sequence):
            control_qubits = list(args[0])
        else:
            control_qubits = list(args)
        if len(control_qubits) != self.ncontrols:
            raise ValueError(f"require {self.ncontrols} control qubits, but received {len(control_qubits)}")
        self.control_qubits = control_qubits
        # enable chaining
        return self

    def _circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if len(self.control_qubits) != self.ncontrols:
            raise RuntimeError("unspecified control qubit(s)")
        tprtcl = self.tgate.particles()
        if len(tprtcl) != self.tgate.num_wires:
            raise RuntimeError("unspecified target gate particle(s)")
        # the ordering of control qubits is irrelevant,
        # so we sort indices to avoid unncessary wire permutations
        icwire = sorted([_map_particle_to_wire(fields, q) for q in self.control_qubits])
        itwire = [_map_particle_to_wire(fields, p) for p in tprtcl]
        iwire = itwire + icwire     # target wires come first
        if any([iw < 0 for iw in iwire]):
            raise RuntimeError("particle not found among fields")
        nwires = sum([f.lattice.nsites for f in fields])
        return _distribute_to_wires(nwires, iwire, csr_matrix(self.as_matrix()))


class TimeEvolutionGate(Gate):
    """
    Quantum time evolution gate, i.e., matrix exponential,
    given a field operator (Hamiltonian) `h`:
    .. math:: e^{-i h t}
    """
    def __init__(self, h: FieldOperator, t: float):
        self.h = h
        self.t = t

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        # might be Hermitian in special cases, but difficult to check,
        # so returning False here for simplicity
        return False

    def as_matrix(self):
        """
        Generate the matrix representation of the time evolution gate.
        """
        hmat = self.h.as_matrix().toarray()
        # TODO: exploit that `hmat` is Hermitian for computing the matrix exponential
        return expm(-1j*self.t * hmat)

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        fields = self.h.fields()
        return sum(f.lattice.nsites for f in fields)

    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        fields = self.h.fields()
        prtcl = []
        for f in fields:
            prtcl = prtcl + [Particle(f, i) for i in range(f.lattice.nsites)]
        return prtcl

    def fields(self):
        """
        Return the list fields hosting the quantum particles which the gate acts on.
        """
        return self.h.fields()

    def _circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        prtcl = self.particles()
        assert len(prtcl) == self.num_wires
        iwire = [_map_particle_to_wire(fields, p) for p in prtcl]
        if any([iw < 0 for iw in iwire]):
            raise RuntimeError("particle not found among fields")
        nwires = sum([f.lattice.nsites for f in fields])
        return _distribute_to_wires(nwires, iwire, csr_matrix(self.as_matrix()))


def _map_particle_to_wire(fields: Sequence[Field], p: Particle):
    """
    Map a particle to a quantum wire.
    """
    i = 0
    for f in fields:
        if p.field == f:
            i += p.index
            return i
        else:
            i += f.lattice.nsites
    # not found
    return -1


def _distribute_to_wires(nwires: int, iwire, gmat: csr_matrix):
    """
    Sparse matrix representation of a quantum gate
    acting on quantum wires in `iwire`.

    Currently assumes that each wire has local dimension 2.
    """
    # complementary wires
    iwcompl = list(set(range(nwires)).difference(iwire))
    assert len(iwire) + len(iwcompl) == nwires

    m = len(iwire)
    assert m <= nwires
    assert gmat.shape == (2**m, 2**m)

    values = np.zeros(2**(nwires - m) * gmat.nnz, dtype=gmat.dtype)
    rowind = np.zeros_like(values, dtype=int)
    colind = np.zeros_like(values, dtype=int)

    for j in range(2**m):
        r = 0
        for b in range(m):
            if j & (1 << b):
                r += (1 << iwire[b])
        for i in range(gmat.indptr[j], gmat.indptr[j + 1]):
            c = 0
            for b in range(m):
                if gmat.indices[i] & (1 << b):
                    c += (1 << iwire[b])
            rowind[i] = r
            colind[i] = c
    values[:gmat.nnz] = gmat.data

    # copy values (corresponds to Kronecker product with identity)
    for k in range(1, 2**(nwires - m)):
        koffset = 0
        for b in range(nwires - m):
            if k & (1 << b):
                koffset += (1 << iwcompl[b])
        rowind[gmat.nnz*k:gmat.nnz*(k+1)] = rowind[:gmat.nnz] + koffset
        colind[gmat.nnz*k:gmat.nnz*(k+1)] = colind[:gmat.nnz] + koffset
        values[gmat.nnz*k:gmat.nnz*(k+1)] = gmat.data

    return csr_matrix((values, (rowind, colind)), shape=(2**nwires, 2**nwires))
