import abc
from enum import Enum
from copy import copy
from typing import Sequence
import numpy as np
from scipy.linalg import expm, sqrtm, block_diag
from scipy.sparse import csr_matrix
from qib.field import Field, Particle, Qubit
from qib.operator import AbstractOperator
from qib.tensor_network import SymbolicTensor, SymbolicBond, SymbolicTensorNetwork, TensorNetwork
from qib.util import map_particle_to_wire


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

    @abc.abstractmethod
    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """

    @abc.abstractmethod
    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """

    @abc.abstractmethod
    def inverse(self):
        """
        Return the inverse operator.
        """

    @abc.abstractmethod
    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """

    @abc.abstractmethod
    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate,
        using an individual tensor axis for each wire.
        """

    @abc.abstractmethod
    def __copy__(self):
        """
        Create a copy of the gate.
        """

    @abc.abstractmethod
    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """


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
        return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        return []

    def inverse(self):
        """
        Return the inverse operator.
        """
        return self

    def on(self, qubit: Qubit):
        """
        Act on the specified qubit.
        """
        self.qubit = qubit
        # enable chaining
        return self

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.qubit:
            raise RuntimeError("unspecified target qubit")
        iwire = map_particle_to_wire(fields, self.qubit)
        if iwire < 0:
            raise RuntimeError("qubit not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, [iwire], csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        return TensorNetwork.wrap(self.as_matrix(), "PauliX")

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        return PauliXGate(self.qubit)

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        if type(other) == type(self) and other.qubit == self.qubit:
            return True
        return False


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
        return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        return []

    def inverse(self):
        """
        Return the inverse operator.
        """
        return self

    def on(self, qubit: Qubit):
        """
        Act on the specified qubit.
        """
        self.qubit = qubit
        # enable chaining
        return self

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.qubit:
            raise RuntimeError("unspecified target qubit")
        iwire = map_particle_to_wire(fields, self.qubit)
        if iwire < 0:
            raise RuntimeError("qubit not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, [iwire], csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        return TensorNetwork.wrap(self.as_matrix(), "PauliY")

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        return PauliYGate(self.qubit)

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return type(other) == type(self) and other.qubit == self.qubit


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
        return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        return []

    def inverse(self):
        """
        Return the inverse operator.
        """
        return self

    def on(self, qubit: Qubit):
        """
        Act on the specified qubit.
        """
        self.qubit = qubit
        # enable chaining
        return self

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.qubit:
            raise RuntimeError("unspecified target qubit")
        iwire = map_particle_to_wire(fields, self.qubit)
        if iwire < 0:
            raise RuntimeError("qubit not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, [iwire], csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        return TensorNetwork.wrap(self.as_matrix(), "PauliZ")

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        return PauliZGate(self.qubit)

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return type(other) == type(self) and other.qubit == self.qubit


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
        return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        return []

    def inverse(self):
        """
        Return the inverse operator.
        """
        return self

    def on(self, qubit: Qubit):
        """
        Act on the specified qubit.
        """
        self.qubit = qubit
        # enable chaining
        return self

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.qubit:
            raise RuntimeError("unspecified target qubit")
        iwire = map_particle_to_wire(fields, self.qubit)
        if iwire < 0:
            raise RuntimeError("qubit not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, [iwire], csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        return TensorNetwork.wrap(self.as_matrix(), "Hadamard")

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        return HadamardGate(self.qubit)

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return type(other) == type(self) and other.qubit == self.qubit


class RxGate(Gate):
    """
    X-axis rotation gate.
    """
    def __init__(self, theta: float, qubit: Qubit=None):
        self.theta = theta
        self.qubit = qubit

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return False

    def as_matrix(self):
        """
        Generate the matrix representation of the gate.
        """
        c = np.cos(self.theta/2)
        s = np.sin(self.theta/2)
        return np.array([[c, -1j*s], [-1j*s, c]])

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return 1

    @property
    def rotation_angle(self):
        """
        The rotation angle
        """
        return self.theta

    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        if self.qubit:
            return [self.qubit]
        return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        return []

    def inverse(self):
        """
        Return the inverse operator.
        """
        return RxGate(-self.theta, self.qubit)

    def on(self, qubit: Qubit):
        """
        Act on the specified qubit.
        """
        self.qubit = qubit
        # enable chaining
        return self

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.qubit:
            raise RuntimeError("unspecified target qubit")
        iwire = map_particle_to_wire(fields, self.qubit)
        if iwire < 0:
            raise RuntimeError("qubit not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, [iwire], csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        # require a unique name for each rotation angle
        return TensorNetwork.wrap(self.as_matrix(), f"Rx({self.theta})")

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        return RxGate(self.theta, self.qubit)

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return type(other) == type(self) and other.qubit == self.qubit and other.theta == self.theta


class RyGate(Gate):
    """
    Y-axis rotation gate
    """
    def __init__(self, theta: float, qubit: Qubit=None):
        self.theta = theta
        self.qubit = qubit

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return False

    def as_matrix(self):
        """
        Generate the matrix representation of the gate.
        """
        c = np.cos(self.theta/2)
        s = np.sin(self.theta/2)
        return np.array([[c, -s], [s, c]])

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return 1

    @property
    def rotation_angle(self):
        """
        The rotation angle
        """
        return self.theta

    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        if self.qubit:
            return [self.qubit]
        return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        return []

    def inverse(self):
        """
        Return the inverse operator.
        """
        return RyGate(-self.theta, self.qubit)

    def on(self, qubit: Qubit):
        """
        Act on the specified qubit.
        """
        self.qubit = qubit
        # enable chaining
        return self

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.qubit:
            raise RuntimeError("unspecified target qubit")
        iwire = map_particle_to_wire(fields, self.qubit)
        if iwire < 0:
            raise RuntimeError("qubit not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, [iwire], csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        # require a unique name for each rotation angle
        return TensorNetwork.wrap(self.as_matrix(), f"Ry({self.theta})")

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        return RyGate(self.theta, self.qubit)

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return type(other) == type(self) and other.qubit == self.qubit and other.theta == self.theta


class RzGate(Gate):
    """
    Z-axis rotation gate
    """
    def __init__(self, theta: float, qubit: Qubit=None):
        self.theta = theta
        self.qubit = qubit

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return False

    def as_matrix(self):
        """
        Generate the matrix representation of the gate.
        """
        x = np.exp(1j*self.theta/2)
        return np.array([[x.conj(), 0], [0, x]])

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return 1

    @property
    def rotation_angle(self):
        """
        The rotation angle
        """
        return self.theta

    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        if self.qubit:
            return [self.qubit]
        return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        return []

    def inverse(self):
        """
        Return the inverse operator.
        """
        return RzGate(-self.theta, self.qubit)

    def on(self, qubit: Qubit):
        """
        Act on the specified qubit.
        """
        self.qubit = qubit
        # enable chaining
        return self

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.qubit:
            raise RuntimeError("unspecified target qubit")
        iwire = map_particle_to_wire(fields, self.qubit)
        if iwire < 0:
            raise RuntimeError("qubit not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, [iwire], csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        # require a unique name for each rotation angle
        return TensorNetwork.wrap(self.as_matrix(), f"Rz({self.theta})")

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        return RzGate(self.theta, self.qubit)

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return type(other) == type(self) and other.qubit == self.qubit and other.theta == self.theta


class RotationGate(Gate):
    """
    General rotation gate; the rotation angle and axis are combined into
    a single vector `ntheta` of length 3.
    """
    def __init__(self, ntheta: Sequence[float], qubit: Qubit=None):
        self.ntheta = np.array(ntheta, copy=False)
        if self.ntheta.shape != (3,):
            raise ValueError("'ntheta' must be a vector of length 3")
        self.qubit = qubit

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return False

    def as_matrix(self):
        """
        Generate the matrix representation of the gate.
        """
        theta = np.linalg.norm(self.ntheta)
        if theta == 0:
            return np.identity(2)
        n = self.ntheta / theta
        return (     np.cos(theta/2)*np.identity(2)
                - 1j*np.sin(theta/2)*np.array([[n[2], n[0] - 1j*n[1]], [n[0] + 1j*n[1], -n[2]]]))

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
        return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        return []

    def inverse(self):
        """
        Return the inverse operator.
        """
        return RotationGate(-self.ntheta, self.qubit)

    def on(self, qubit: Qubit):
        """
        Act on the specified qubit.
        """
        self.qubit = qubit
        # enable chaining
        return self

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.qubit:
            raise RuntimeError("unspecified target qubit")
        iwire = map_particle_to_wire(fields, self.qubit)
        if iwire < 0:
            raise RuntimeError("qubit not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, [iwire], csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        # require a unique name for each rotation angle
        return TensorNetwork.wrap(self.as_matrix(), f"Rn({self.ntheta})")

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        return RotationGate(self.ntheta, self.qubit)

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return type(other) == type(self) and other.qubit == self.qubit and np.allclose(other.ntheta, self.ntheta)


class SGate(Gate):
    """
    S (phase) gate - provides a phase shift of pi/2.
    """
    def __init__(self, qubit: Qubit=None):
        self.qubit = qubit

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return False

    def as_matrix(self):
        """
        Generate the matrix representation of the gate.
        """
        return np.array([[ 1.,  0.], [ 0.,  1j]])

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
        return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        return []

    def inverse(self):
        """
        Return the inverse operator.
        """
        return SAdjGate(self.qubit)

    def on(self, qubit: Qubit):
        """
        Act on the specified qubit.
        """
        self.qubit = qubit
        # enable chaining
        return self

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.qubit:
            raise RuntimeError("unspecified target qubit")
        iwire = map_particle_to_wire(fields, self.qubit)
        if iwire < 0:
            raise RuntimeError("qubit not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, [iwire], csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        return TensorNetwork.wrap(self.as_matrix(), "S")

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        return SGate(self.qubit)

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return type(other) == type(self) and other.qubit == self.qubit


class SAdjGate(Gate):
    """
    Adjoint of S gate - provides a phase shift of -pi/2.
    """
    def __init__(self, qubit: Qubit=None):
        self.qubit = qubit

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return False

    def as_matrix(self):
        """
        Generate the matrix representation of the gate.
        """
        return np.array([[ 1.,  0.], [ 0., -1j]])

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
        return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        return []

    def inverse(self):
        """
        Return the inverse operator.
        """
        return SGate(self.qubit)

    def on(self, qubit: Qubit):
        """
        Act on the specified qubit.
        """
        self.qubit = qubit
        # enable chaining
        return self

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.qubit:
            raise RuntimeError("unspecified target qubit")
        iwire = map_particle_to_wire(fields, self.qubit)
        if iwire < 0:
            raise RuntimeError("qubit not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, [iwire], csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        return TensorNetwork.wrap(self.as_matrix(), "Sadj")

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        return SAdjGate(self.qubit)

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return type(other) == type(self) and other.qubit == self.qubit


class TGate(Gate):
    """
    T gate - provides a phase shift of pi/4.
    """
    def __init__(self, qubit: Qubit=None):
        self.qubit = qubit

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return False

    def as_matrix(self):
        """
        Generate the matrix representation of the gate.
        """
        return np.array([[ 1.,  0.], [ 0., (1+1j)/np.sqrt(2)]])

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
        return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        return []

    def inverse(self):
        """
        Return the inverse operator.
        """
        return TAdjGate(self.qubit)

    def on(self, qubit: Qubit):
        """
        Act on the specified qubit.
        """
        self.qubit = qubit
        # enable chaining
        return self

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.qubit:
            raise RuntimeError("unspecified target qubit")
        iwire = map_particle_to_wire(fields, self.qubit)
        if iwire < 0:
            raise RuntimeError("qubit not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, [iwire], csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        return TensorNetwork.wrap(self.as_matrix(), "T")

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        return TGate(self.qubit)

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return type(other) == type(self) and other.qubit == self.qubit


class TAdjGate(Gate):
    """
    Adjoint of T gate - provides a phase shift of -pi/4.
    """
    def __init__(self, qubit: Qubit=None):
        self.qubit = qubit

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return False

    def as_matrix(self):
        """
        Generate the matrix representation of the gate.
        """
        return np.array([[ 1.,  0.], [ 0., (1-1j)/np.sqrt(2)]])

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
        return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        return []

    def inverse(self):
        """
        Return the inverse operator.
        """
        return TGate(self.qubit)

    def on(self, qubit: Qubit):
        """
        Act on the specified qubit.
        """
        self.qubit = qubit
        # enable chaining
        return self

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.qubit:
            raise RuntimeError("unspecified target qubit")
        iwire = map_particle_to_wire(fields, self.qubit)
        if iwire < 0:
            raise RuntimeError("qubit not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, [iwire], csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        return TensorNetwork.wrap(self.as_matrix(), "Tadj")

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        return TAdjGate(self.qubit)

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return type(other) == type(self) and other.qubit == self.qubit


class PhaseFactorGate(Gate):
    """
    Phase factor gate: multiplication by the phase factor :math:`e^{i \phi}`.
    """
    def __init__(self, phi: float, nwires: int):
        self.phi = phi
        self.nwires = nwires
        self.prtcl = []

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return False

    def as_matrix(self):
        """
        Generate the matrix representation of the phase factor gate.
        """
        # TODO: generalize base 2
        return np.exp(1j*self.phi) * np.identity(2**self.nwires)

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return self.nwires

    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        return self.prtcl

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        flist = []
        for p in self.prtcl:
            if p.field not in flist:
                flist.append(p.field)
        return flist

    def inverse(self):
        """
        Return the inverse operator.
        """
        return PhaseFactorGate(-self.phi, self.nwires)

    def on(self, *args):
        """
        Act on the specified particle(s).
        """
        if len(args) == 1 and isinstance(args[0], Sequence):
            prtcl = list(args[0])
        else:
            prtcl = list(args)
        if len(prtcl) != self.nwires:
            raise ValueError(f"require {self.nwires} particles, but received {len(prtcl)}")
        self.prtcl = prtcl
        # enable chaining
        return self

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.prtcl:
            raise RuntimeError("unspecified target particle(s)")
        iwire = [map_particle_to_wire(fields, p) for p in self.prtcl]
        if any(iw < 0 for iw in iwire):
            raise RuntimeError("particle not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, iwire, csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        stn = SymbolicTensorNetwork()
        dataref = f"Phase({self.phi / self.nwires})"
        for i in range(self.nwires):
            stn.add_tensor(SymbolicTensor(i, (2, 2), (2*i, 2*i + 1), dataref))
        # virtual tensor for open axes
        stn.add_tensor(SymbolicTensor(-1, 2*self.nwires * (2,),
                                        list(range(0, 2*self.nwires, 2))
                                      + list(range(1, 2*self.nwires, 2)), None))
        for i in range(self.nwires):
            stn.add_bond(SymbolicBond(2*i,     (-1, i)))
            stn.add_bond(SymbolicBond(2*i + 1, (-1, i)))
        assert stn.is_consistent()
        return TensorNetwork(stn, { dataref: np.exp(1j*self.phi / self.nwires) * np.identity(2) })

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        gate = PhaseFactorGate(self.phi, self.nwires)
        gate.on(self.prtcl)
        return gate

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return type(other) == type(self) and other.prtcl == self.prtcl and other.phi == self.phi and other.nwires == self.nwires


class PrepareGate(Gate):
    """
    Vector "preparation" gate.
    """
    def __init__(self, vec, nqubits: int, transpose=False):
        vec = np.array(vec)
        if vec.ndim != 1:
            raise ValueError("expecting a vector")
        if not np.isrealobj(vec):
            raise ValueError("only real-valued vectors supported")
        if vec.shape[0] != 2**nqubits:
            raise ValueError(f"input vector must have length 2^nqubits = {2**nqubits}")
        # note: using 1-norm here by convention
        n = np.linalg.norm(vec, ord=1)
        if abs(n - 1) > 1e-12:
            vec /= n
        self.vec = vec
        self.nqubits = nqubits
        self.qubits = []
        self.transpose = transpose

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        # in general not Hermitian
        # TODO: can one construct a Hermitian realization?
        return False

    def as_matrix(self):
        """
        Generate the matrix representation of the "preparation" gate.
        """
        x = np.sign(self.vec) * np.sqrt(np.abs(self.vec))
        # use QR decomposition for extension to full basis
        Q = np.linalg.qr(x.reshape((-1, 1)), mode="complete")[0]
        if np.dot(x, Q[:, 0]) < 0:
            Q[:, 0] = -Q[:, 0]
        if self.transpose:
            return Q.T
        return Q

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return self.nqubits

    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        return self.qubits

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        flist = []
        for q in self.qubits:
            if q.field not in flist:
                flist.append(q.field)
        return flist

    def inverse(self):
        """
        Return the inverse operator.
        """
        invgate = PrepareGate(self.vec, self.nqubits, not self.transpose)
        if self.qubits:
            invgate.on(self.qubits)
        return invgate

    def on(self, *args):
        """
        Act on the specified qubit(s).
        """
        if len(args) == 1 and isinstance(args[0], Sequence):
            qubits = list(args[0])
        else:
            qubits = list(args)
        if len(qubits) != self.nqubits:
            raise ValueError(f"expecting {self.nqubits} qubits, but received {len(qubits)}")
        self.qubits = qubits
        # enable chaining
        return self

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if len(self.qubits) != self.nqubits:
            raise RuntimeError("unspecified qubit(s)")
        prtcl = self.particles()
        iwire = [map_particle_to_wire(fields, p) for p in prtcl]
        if any(iw < 0 for iw in iwire):
            raise RuntimeError("particle not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, iwire, csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        # construct tensor network corresponding to |vec/norm(vec)> <0|...<0|
        x = np.reshape(np.sign(self.vec) * np.sqrt(np.abs(self.vec)), self.nqubits * (2,))
        stn = SymbolicTensorNetwork()
        xten = SymbolicTensor(0, x.shape, range(self.nqubits), str(hash(x.data.tobytes())))
        stn.add_tensor(xten)
        for i in range(self.nqubits):
            stn.add_tensor(SymbolicTensor(1 + i, (2,), (self.nqubits + i,), "|0>_2"))
        # virtual tensor for open axes
        stn.add_tensor(SymbolicTensor(-1, 2*self.nqubits * (2,),
                                      range(2*self.nqubits) if not self.transpose else
                                      list(range(self.nqubits, 2*self.nqubits)) + list(range(self.nqubits)), None))
        # add bonds to specify open axes
        for i in range(self.nqubits):
            stn.add_bond(SymbolicBond(i, (-1, 0)))
        for i in range(self.nqubits):
            stn.add_bond(SymbolicBond(self.nqubits + i, (-1, 1 + i)))
        assert stn.is_consistent()
        return TensorNetwork(stn, { xten.dataref: x, "|0>_2": np.array([ 1., 0.]) })

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        gate = PrepareGate(self.vec, self.nqubits, self.transpose)
        gate.on(self.qubits)
        return gate

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return (type(other) == type(self) and other.nqubits == self.nqubits and np.allclose(other.vec, self.vec)
                and other.qubits == self.qubits and other.transpose == self.transpose)


class ControlledGate(Gate):
    """
    A controlled quantum gate with an arbitrary number of control qubits.
    The control qubits have to be set explicitly.
    The target qubits (or particles) are specified via the target gate.
    """
    def __init__(self, tgate: Gate, ncontrols: int, ctrl_state: Sequence[int]=None):
        self.tgate = tgate
        self.ncontrols = ncontrols
        self.control_qubits = []
        # standard case: control is active if all control qubits are in |1> state
        if ctrl_state is None:
            ctrl_state = ncontrols * [1]
        if len(ctrl_state) != ncontrols:
            raise ValueError(f"length of `ctrl_state` must be equal to number of control qubits, {ncontrols}")
        for i in ctrl_state:
            if not i in (0, 1):
                raise ValueError(f"`ctrl_state` can only contain 0 or 1 bits, received {i}")
        self.ctrl_state = list(ctrl_state)

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return self.tgate.is_hermitian()

    def inverse(self):
        """
        Return the inverse operator.
        """
        invgate = ControlledGate(self.tgate.inverse(), self.ncontrols, self.ctrl_state)
        if self.control_qubits:
            invgate.set_control(self.control_qubits)
        return invgate

    def as_matrix(self):
        """
        Generate the matrix representation of the controlled gate.
        Format: |control> x |target>
        """
        tgmat = self.tgate.as_matrix()
        # target gate corresponds to faster varying indices
        assert self.ncontrols == len(self.ctrl_state)
        cidx = np.zeros(2**self.ncontrols)
        ic = 0
        for j in range(self.ncontrols):
            if self.ctrl_state[j] == 1:
                # first digit is the most significant bit
                ic += 1 << (self.ncontrols - 1 - j)
        cidx[ic] = 1
        return (  np.kron(np.diag(1 - cidx), np.identity(tgmat.shape[0]))
                + np.kron(np.diag(cidx), tgmat))

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
        return self.control_qubits + self.tgate.particles()

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        flist = []
        for q in self.control_qubits:
            if q.field not in flist:
                flist.append(q.field)
        for f in self.tgate.fields():
            if f not in flist:
                flist.append(f)
        return flist

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

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if len(self.control_qubits) != self.ncontrols:
            raise RuntimeError("unspecified control qubit(s)")
        if len(self.tgate.particles()) != self.tgate.num_wires:
            raise RuntimeError("unspecified target gate particle(s)")
        prtcl = self.particles()
        iwire = [map_particle_to_wire(fields, p) for p in prtcl]
        if any(iw < 0 for iw in iwire):
            raise RuntimeError("particle not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, iwire, csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        if isinstance(self.tgate, ControlledGate):
            # in case target gate is also a controlled gate...
            return ControlledGate(self.tgate.tgate, self.ncontrols + self.tgate.ncontrols,
                                  self.ctrl_state + self.tgate.ctrl_state).as_tensornet()
        # use matrix representation of target gate in tensor network, for simplicity
        ntargets = self.tgate.num_wires
        ctgmat = np.reshape(self.tgate.as_matrix(), 2*ntargets * (2,))
        # elevate target gate tensor to a controlled tensor
        ctgmat = np.stack((np.reshape(np.identity(2**ntargets), 2*ntargets * (2,)),
                           ctgmat), axis=0)
        stn = SymbolicTensorNetwork()
        # dummy bond IDs will be set later
        ctgten = SymbolicTensor(0, ctgmat.shape, ctgmat.ndim * [-1], "ctrl_" + str(hash(ctgmat.data.tobytes())))
        stn.add_tensor(ctgten)
        data = { ctgten.dataref: ctgmat }
        # virtual tensor for open axes
        oaxten = SymbolicTensor(-1, 2*(self.ncontrols + ntargets) * (2,), 2*(self.ncontrols + ntargets) * [-1], None)
        stn.add_tensor(oaxten)
        # next available bond ID
        bid_next = 0
        # add bonds to specify open target gate axes
        for i in range(ntargets):
            stn.add_bond(SymbolicBond(bid_next, (-1, 0)))
            ctgten.bids[1 + i] = bid_next
            oaxten.bids[self.ncontrols + i] = bid_next
            bid_next += 1
        for i in range(ntargets):
            stn.add_bond(SymbolicBond(bid_next, (-1, 0)))
            ctgten.bids[1 + ntargets + i] = bid_next
            oaxten.bids[2*self.ncontrols + ntargets + i] = bid_next
            bid_next += 1
        # "wire crossing" tensors:
        # axis ordering: physical output wire, physical input wire, upward axis, downward axis
        # control remains active if in |1> state
        ctrl_cross_pos = np.zeros(shape=(2, 2, 2, 2))
        ctrl_cross_pos[0, 0, 0, 0] = 1
        ctrl_cross_pos[1, 1, 1, 1] = 1
        ctrl_cross_pos[0, 0, 1, 0] = 1
        ctrl_cross_pos[1, 1, 0, 0] = 1
        # control remains active if in |0> state
        ctrl_cross_neg = np.zeros(shape=(2, 2, 2, 2))
        ctrl_cross_neg[1, 1, 0, 0] = 1
        ctrl_cross_neg[0, 0, 1, 1] = 1
        ctrl_cross_neg[1, 1, 1, 0] = 1
        ctrl_cross_neg[0, 0, 0, 0] = 1
        ctrl_cross = [ctrl_cross_neg, ctrl_cross_pos]
        cc_dataref = ["ctrl_cross_neg", "ctrl_cross_pos"]
        if self.ctrl_state[0] == 0:
            # insert Pauli-X gates for negated control
            x_ten1 = SymbolicTensor(self.ncontrols, (2, 2), [bid_next, -1], "PauliX")
            stn.add_tensor(x_ten1)
            stn.add_bond(SymbolicBond(bid_next, (-1, x_ten1.tid)))
            oaxten.bids[0] = bid_next
            bid_next += 1
            x_ten2 = SymbolicTensor(self.ncontrols + 1, (2, 2), [-1, bid_next], "PauliX")
            stn.add_tensor(x_ten2)
            stn.add_bond(SymbolicBond(bid_next, (-1, x_ten2.tid)))
            oaxten.bids[self.ncontrols + ntargets] = bid_next
            bid_next += 1
            data["PauliX"] = np.array([[ 0.,  1.], [ 1.,  0.]])
        for i in range(1, self.ncontrols):
            j = self.ctrl_state[i]
            ccrten = SymbolicTensor(i, ctrl_cross[j].shape, [bid_next, bid_next + 1, -1, -1], cc_dataref[j])
            stn.add_tensor(ccrten)
            stn.add_bond(SymbolicBond(bid_next, (-1, i)))
            oaxten.bids[i] = bid_next
            bid_next += 1
            stn.add_bond(SymbolicBond(bid_next, (-1, i)))
            oaxten.bids[self.ncontrols + ntargets + i] = bid_next
            bid_next += 1
            if i == 1:
                if self.ctrl_state[0] == 1:
                    stn.add_bond(SymbolicBond(bid_next, (-1, -1, i)))
                    oaxten.bids[0] = bid_next
                    oaxten.bids[self.ncontrols + ntargets] = bid_next
                    ccrten.bids[2] = bid_next
                    bid_next += 1
                else:
                    # connect to Pauli-X gates
                    stn.add_bond(SymbolicBond(bid_next, (self.ncontrols, self.ncontrols + 1, i)))
                    x_ten1.bids[1] = bid_next
                    x_ten2.bids[0] = bid_next
                    ccrten.bids[2] = bid_next
                    bid_next += 1
            else:
                # vertical control bond connection
                stn.add_bond(SymbolicBond(bid_next, (i - 1, i)))
                stn.get_tensor(i - 1).bids[3] = bid_next
                ccrten.bids[2] = bid_next
                bid_next += 1
            if cc_dataref[j] not in data:
                data[cc_dataref[j]] = ctrl_cross[j]
        # control bond connected to target gate tensor
        if self.ncontrols == 1:
            if self.ctrl_state[0] == 1:
                stn.add_bond(SymbolicBond(bid_next, (0, -1, -1)))
                ctgten.bids[0] = bid_next
                oaxten.bids[0] = bid_next
                oaxten.bids[self.ncontrols + ntargets] = bid_next
                bid_next += 1
            else:
                # connect to Pauli-X gates
                stn.add_bond(SymbolicBond(bid_next, (0, 1, 2)))
                ctgten.bids[0] = bid_next
                x_ten1.bids[1] = bid_next
                x_ten2.bids[0] = bid_next
                bid_next += 1
        else:   # self.ncontrols > 1
            stn.add_bond(SymbolicBond(bid_next, (0, self.ncontrols - 1)))
            ctgten.bids[0] = bid_next
            stn.get_tensor(self.ncontrols - 1).bids[3] = bid_next
            bid_next += 1
        assert stn.is_consistent()
        return TensorNetwork(stn, data)

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        gate = ControlledGate(copy(self.tgate), self.ncontrols, self.ctrl_state)
        gate.set_control(self.control_qubits)
        return gate

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return (type(other) == type(self)
                and other.tgate == self.tgate
                and other.ncontrols == self.ncontrols
                and other.ctrl_state == self.ctrl_state
                and other.control_qubits == self.control_qubits)


class RxxGate(Gate):
    """
    Rxx gate.
    """
    def __init__(self, theta: float, q1: Gate, q2 : Gate):
        self.theta = theta
        self.q1 = q1
        self.q2 = q2
        
    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        # in general not Hermitian unless theta = 0
        return False

    def inverse(self):
        """
        Return the inverse operator.
        """
        invgate = RxxGate(-self.theta, self.q1, self.q2)
        return invgate

    def as_matrix(self):
        """
        Generate the matrix representation of the gate.
        """
        x = np.cos(self.theta/2)
        y = -1j*np.sin(self.theta/2)
        return np.array([
            [x, 0, 0, y],
            [0, x, y, 0],
            [0, y, x, 0],
            [y, 0, 0, x]
        ])

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return 2

    @property
    def rotation_angle(self):
        """
        The rotation angle
        """
        return self.theta

    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        return [self.q2, self.q1]

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        flist = []
        if self.q1.field not in flist:
            flist.append(self.q1.field)
        if self.q2.field not in flist:
            flist.append(self.q2.field)
        return flist

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.q2:
            raise RuntimeError("unspecified control qubit")
        if not self.q1:
            raise RuntimeError("unspecified target qubit")
        prtcl = self.particles()
        iwire = [map_particle_to_wire(fields, p) for p in prtcl]
        if any(iw < 0 for iw in iwire):
            raise RuntimeError("particle not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, iwire, csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        return TensorNetwork.wrap(self.as_matrix(), f"Rxx({self.theta})")

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        gate = RxxGate(self.theta, copy(self.q1), copy(self.q2))
        return gate

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return (type(other) == type(self)
                and other.theta == self.theta
                and other.q1 == self.q1
                and other.q2 == self.q2)


class RyyGate(Gate):
    """
    Ryy gate.
    """
    def __init__(self, theta: float, q1: Gate, q2 : Gate):
        self.theta = theta
        self.q1 = q1
        self.q2 = q2
        
    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        # in general not Hermitian unless theta = 0
        return False

    def inverse(self):
        """
        Return the inverse operator.
        """
        invgate = RyyGate(-self.theta, self.q1, self.q2)
        return invgate

    def as_matrix(self):
        """
        Generate the matrix representation of the gate.
        """
        x = np.cos(self.theta/2)
        y = 1j*np.sin(self.theta/2)
        return np.array([
            [x,  0,  0,  y],
            [0,  x, -y,  0],
            [0, -y,  x,  0],
            [y,  0,  0,  x]
        ])

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return 2

    @property
    def rotation_angle(self):
        """
        The rotation angle
        """
        return self.theta

    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        return [self.q2, self.q1]

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        flist = []
        if self.q1.field not in flist:
            flist.append(self.q1.field)
        if self.q2.field not in flist:
            flist.append(self.q2.field)
        return flist

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.q2:
            raise RuntimeError("unspecified control qubit")
        if not self.q1:
            raise RuntimeError("unspecified target qubit")
        prtcl = self.particles()
        iwire = [map_particle_to_wire(fields, p) for p in prtcl]
        if any(iw < 0 for iw in iwire):
            raise RuntimeError("particle not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, iwire, csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        return TensorNetwork.wrap(self.as_matrix(), f"Ryy({self.theta})")

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        gate = RyyGate(self.theta, copy(self.q1), copy(self.q2))
        return gate

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return (type(other) == type(self)
                and other.theta == self.theta
                and other.q1 == self.q1
                and other.q2 == self.q2)


class RzzGate(Gate):
    """
    Rzz gate.
    """
    def __init__(self, theta: float, q1: Gate, q2 : Gate):
        self.theta = theta
        self.q1 = q1
        self.q2 = q2
        
    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        # in general not Hermitian unless theta = 0
        return False

    def inverse(self):
        """
        Return the inverse operator.
        """
        invgate = RzzGate(-self.theta, self.q1, self.q2)
        return invgate

    def as_matrix(self):
        """
        Generate the matrix representation of the gate.
        """
        x = np.exp(-1j*self.theta/2)
        y = x.conj()
        return np.array([
            [x, 0, 0, 0],
            [0, y, 0, 0],
            [0, 0, y, 0],
            [0, 0, 0, x]
        ])

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return 2

    @property
    def rotation_angle(self):
        """
        The rotation angle
        """
        return self.theta

    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        return [self.q2, self.q1]

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        flist = []
        if self.q1.field not in flist:
            flist.append(self.q1.field)
        if self.q2.field not in flist:
            flist.append(self.q2.field)
        return flist

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.q2:
            raise RuntimeError("unspecified control qubit")
        if not self.q1:
            raise RuntimeError("unspecified target qubit")
        prtcl = self.particles()
        iwire = [map_particle_to_wire(fields, p) for p in prtcl]
        if any(iw < 0 for iw in iwire):
            raise RuntimeError("particle not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, iwire, csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        return TensorNetwork.wrap(self.as_matrix(), f"Rzz({self.theta})")

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        gate = RzzGate(self.theta, copy(self.q1), copy(self.q2))
        return gate

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return (type(other) == type(self)
                and other.theta == self.theta
                and other.q1 == self.q1
                and other.q2 == self.q2)


class MultiplexedGate(Gate):
    """
    Multiplexed gate (control qubits select a unitary), generalizing a controlled gate.
    """
    def __init__(self, tgates: Sequence[Gate], ncontrols: int):
        if len(tgates) != 2**ncontrols:
            assert ValueError(f"require {2**ncontrols} target gates for {ncontrols} control qubits")
        self.tgates = list(tgates)
        self.ncontrols = ncontrols
        self.control_qubits = []

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return all(g.is_hermitian() for g in self.tgates)

    def inverse(self):
        """
        Return the inverse operator.
        """
        invgate = MultiplexedGate([g.inverse() for g in self.tgates], self.ncontrols)
        if self.control_qubits:
            invgate.set_control(self.control_qubits)
        return invgate

    def as_matrix(self):
        """
        Generate the matrix representation of the multiplexed gate.
        """
        tgmat = [g.as_matrix() for g in self.tgates]
        # target gates correspond to faster varying indices
        return block_diag(*tgmat)

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        assert self.tgates
        return self.tgates[0].num_wires + self.ncontrols

    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        assert self.tgates
        tprtcl = self.tgates[0].particles()
        for g in self.tgates:
            assert tprtcl == g.particles(), "particles of all target gates must match"
        return self.control_qubits + tprtcl

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        flist = []
        for q in self.control_qubits:
            if q.field not in flist:
                flist.append(q.field)
        assert self.tgates
        for g in self.tgates:
            assert self.tgates[0].fields() == g.fields(), "fields of all target gates must match"
        for f in self.tgates[0].fields():
            if f not in flist:
                flist.append(f)
        return flist

    def target_gates(self):
        """
        Get the target gates.
        """
        return self.tgates

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

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if len(self.control_qubits) != self.ncontrols:
            raise RuntimeError("unspecified control qubit(s)")
        for g in self.tgates:
            if len(g.particles()) != g.num_wires:
                raise RuntimeError("unspecified target gate particle(s)")
        prtcl = self.particles()
        iwire = [map_particle_to_wire(fields, p) for p in prtcl]
        if any(iw < 0 for iw in iwire):
            raise RuntimeError("particle not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, iwire, csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        ntargets = self.tgates[0].num_wires
        # use matrix representation of target gates in tensor network, for simplicity
        mtgmat = np.reshape(np.stack([g.as_matrix() for g in self.tgates], axis=0), self.ncontrols * (2,) + 2*ntargets * (2,))
        stn = SymbolicTensorNetwork()
        ctgten = SymbolicTensor(0, mtgmat.shape, mtgmat.ndim * [-1], str(hash(mtgmat.data.tobytes())))
        stn.add_tensor(ctgten)
        # virtual tensor for open axes
        oaxten = SymbolicTensor(-1, 2*(self.ncontrols + ntargets) * (2,), 2*(self.ncontrols + ntargets) * [-1], None)
        stn.add_tensor(oaxten)
        # next available bond ID
        bid_next = 0
        # add bonds to specify open target gate axes
        for i in range(ntargets):
            stn.add_bond(SymbolicBond(bid_next, (-1, 0)))
            oaxten.bids[self.ncontrols + i] = bid_next
            ctgten.bids[self.ncontrols + i] = bid_next
            bid_next += 1
        for i in range(ntargets):
            stn.add_bond(SymbolicBond(bid_next, (-1, 0)))
            oaxten.bids[2*self.ncontrols + ntargets + i] = bid_next
            ctgten.bids[self.ncontrols + ntargets + i]   = bid_next
            bid_next += 1
        for i in range(self.ncontrols):
            stn.add_bond(SymbolicBond(bid_next, (-1, -1, 0)))
            oaxten.bids[i]                             = bid_next
            oaxten.bids[self.ncontrols + ntargets + i] = bid_next
            ctgten.bids[i]                             = bid_next
            bid_next += 1
        assert stn.is_consistent()
        return TensorNetwork(stn, { ctgten.dataref: mtgmat })

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        gate = MultiplexedGate(copy(self.tgates), self.ncontrols)
        gate.set_control(self.control_qubits)
        return gate

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return (type(other) == type(self)
                and other.tgates == self.tgates
                and other.ncontrols == self.ncontrols
                and other.control_qubits == self.control_qubits)


class TimeEvolutionGate(Gate):
    """
    Quantum time evolution gate, i.e., matrix exponential,
    given a Hamiltonian `h`: :math:`e^{-i h t}`.
    """
    def __init__(self, h: AbstractOperator, t: float):
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
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        return self.h.fields()

    def inverse(self):
        """
        Return the inverse operator.
        """
        return TimeEvolutionGate(self.h, -self.t)

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        prtcl = self.particles()
        assert len(prtcl) == self.num_wires
        iwire = [map_particle_to_wire(fields, p) for p in prtcl]
        if any(iw < 0 for iw in iwire):
            raise RuntimeError("particle not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, iwire, csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        # local dimensions of all particles
        dims = [p.field.local_dim for p in self.particles()]
        umat = self.as_matrix()
        return TensorNetwork.wrap(np.reshape(umat, dims + dims), str(hash(umat.data.tobytes())))

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        return TimeEvolutionGate(self.h, self.t)

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return type(other) == type(self) and other.h == self.h and other.t == self.t


class BlockEncodingMethod(Enum):
    """
    Block encoding method.
    """
    Wx  = 1
    Wxi = 2 # inverse Wx
    R   = 3


class BlockEncodingGate(Gate):
    """
    Block encoding gate of a Hamiltonian `h`, assumed to be Hermitian
    and normalized such that its spectral norm is bounded by 1.
    Output state is Hamiltonian applied to principal input state
    if auxiliary qubit(s) is initialized to |0>.
    """
    def __init__(self, h: AbstractOperator, method: BlockEncodingMethod = BlockEncodingMethod.Wx):
        self.h = h
        self.method = method
        self.auxiliary_qubits = []

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        if self.method == BlockEncodingMethod.Wx:
            return False
        if self.method == BlockEncodingMethod.Wxi:
            return False
        if self.method == BlockEncodingMethod.R:
            # assuming that `h` is Hermitian
            return True
        raise NotImplementedError(f"encoding method {self.method} not supported yet")

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        fields = self.h.fields()
        return sum(f.lattice.nsites for f in fields) + self.num_aux_qubits

    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        if len(self.auxiliary_qubits) != self.num_aux_qubits:
            raise RuntimeError(f"require {self.num_aux_qubits} auxiliary qubits, but have {len(self.auxiliary_qubits)}")
        prtcl = [q for q in self.auxiliary_qubits]  # copy of list
        fields = self.h.fields()
        for f in fields:
            prtcl += [Particle(f, i) for i in range(f.lattice.nsites)]
        return prtcl

    def inverse(self):
        """
        Return the inverse operator.
        """
        if self.method == BlockEncodingMethod.Wx:
            ginv = BlockEncodingGate(self.h, BlockEncodingMethod.Wxi)
            if self.auxiliary_qubits:
                ginv.set_auxiliary_qubits(self.auxiliary_qubits)
            return ginv
        if self.method == BlockEncodingMethod.Wxi:
            ginv = BlockEncodingGate(self.h, BlockEncodingMethod.Wx)
            if self.auxiliary_qubits:
                ginv.set_auxiliary_qubits(self.auxiliary_qubits)
            return ginv
        if self.method == BlockEncodingMethod.R:
            return self
        raise NotImplementedError(f"encoding method {self.method} not supported yet")

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        flist = []
        for q in self.auxiliary_qubits:
            if q.field not in flist:
                flist.append(q.field)
        for f in self.h.fields():
            if f not in flist:
                flist.append(f)
        return flist

    def encoded_operator(self):
        """
        Get the encoded operator.
        """
        return self.h

    @property
    def num_aux_qubits(self):
        """
        Number of auxiliary qubits.
        """
        if self.method == BlockEncodingMethod.Wx:
            return 1
        if self.method == BlockEncodingMethod.Wxi:
            return 1
        if self.method == BlockEncodingMethod.R:
            return 1
        raise NotImplementedError(f"encoding method {self.method} not supported yet")

    def set_auxiliary_qubits(self, *args):
        """
        Set the auxiliary qubits.
        """
        if len(args) == 1 and isinstance(args[0], Sequence):
            auxiliary_qubits = list(args[0])
        else:
            auxiliary_qubits = list(args)
        if len(auxiliary_qubits) != self.num_aux_qubits:
            raise ValueError(f"require {self.num_aux_qubits} auxiliary qubits, but received {len(auxiliary_qubits)}")
        self.auxiliary_qubits = auxiliary_qubits
        # enable chaining
        return self

    def as_matrix(self):
        """
        Generate the matrix representation of the block encoding gate.
        Format: |ancillary> @ |encoded_state>
        """
        # assuming that `h` is Hermitian and that its spectral norm is bounded by 1
        hmat = self.h.as_matrix().toarray()
        sq1h = sqrtm(np.identity(hmat.shape[0]) - hmat @ hmat)
        if self.method == BlockEncodingMethod.Wx:
            return np.block([[hmat, 1j*sq1h], [1j*sq1h, hmat]])
        if self.method == BlockEncodingMethod.Wxi:
            return np.block([[hmat, -1j*sq1h], [-1j*sq1h, hmat]])
        if self.method == BlockEncodingMethod.R:
            return np.block([[hmat, sq1h], [sq1h, -hmat]])
        raise NotImplementedError(f"encoding method {self.method} not supported yet")

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if len(self.auxiliary_qubits) != self.num_aux_qubits:
            raise RuntimeError("unspecified auxiliary qubit(s)")
        prtcl = self.particles()
        assert len(prtcl) == self.num_wires
        iwire = [map_particle_to_wire(fields, p) for p in prtcl]
        if any(iw < 0 for iw in iwire):
            raise RuntimeError("particle not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, iwire, csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        raise NotImplementedError("as_tensornet not yet implemented for this gate")

    def __copy__(self):
        """
        Copy of the gate
        """
        block = BlockEncodingGate(self.h, self.method)
        if self.auxiliary_qubits:
            block.set_auxiliary_qubits(self.auxiliary_qubits)
        return block

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        # TODO: generalize the equivalence test between 2 generic operators...
        return (type(other) == type(self)
                and other.h == self.h
                and other.method == self.method
                and other.auxiliary_qubits == self.auxiliary_qubits)


class GeneralGate(Gate):
    """
    General (user-defined) quantum gate, specified by a unitary matrix.
    """
    def __init__(self, mat, nwires: int):
        mat = np.array(mat, copy=False)
        if mat.shape != (2**nwires, 2**nwires):
            raise ValueError(f"`mat` must be a {2**nwires} x {2**nwires} matrix")
        if not np.allclose(mat @ mat.conj().T, np.identity(mat.shape[0])):
            raise ValueError("`mat` must be unitary")
        self.mat = mat
        self.nwires = nwires
        self.prtcl = []

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        return np.allclose(self.mat, self.mat.conj().T)

    def as_matrix(self):
        """
        Return the matrix representation of the gate.
        """
        return self.mat

    @property
    def num_wires(self):
        """
        The number of "wires" (or quantum particles) this gate acts on.
        """
        return self.nwires

    def particles(self):
        """
        Return the list of quantum particles the gate acts on.
        """
        return self.prtcl

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        flist = []
        for p in self.prtcl:
            if p.field not in flist:
                flist.append(p.field)
        return flist

    def inverse(self):
        """
        Return the inverse operator.
        """
        return GeneralGate(self.mat.conj().T, self.nwires)

    def on(self, *args):
        """
        Act on the specified particle(s).
        """
        if len(args) == 1 and isinstance(args[0], Sequence):
            prtcl = list(args[0])
        else:
            prtcl = list(args)
        if len(prtcl) != self.nwires:
            raise ValueError(f"require {self.nwires} particles, but received {len(prtcl)}")
        self.prtcl = prtcl
        # enable chaining
        return self

    def as_circuit_matrix(self, fields: Sequence[Field]):
        """
        Generate the sparse matrix representation of the gate
        as element of a quantum circuit.
        """
        for f in fields:
            if f.local_dim != 2:
                raise NotImplementedError("quantum wire indexing assumes local dimension 2")
        if not self.prtcl:
            raise RuntimeError("unspecified target particle(s)")
        iwire = [map_particle_to_wire(fields, p) for p in self.prtcl]
        if any(iw < 0 for iw in iwire):
            raise RuntimeError("particle not found among fields")
        nwires = sum(f.lattice.nsites for f in fields)
        return _distribute_to_wires(nwires, iwire, csr_matrix(self.as_matrix()))

    def as_tensornet(self):
        """
        Generate a tensor network representation of the gate.
        """
        return TensorNetwork.wrap(np.reshape(self.as_matrix(), 2*self.nwires * (2,)), str(hash(self.mat.data.tobytes())))

    def __copy__(self):
        """
        Create a copy of the gate.
        """
        gate = GeneralGate(self.mat, self.nwires)
        gate.on(self.prtcl)
        return gate

    def __eq__(self, other):
        """
        Check if gates are equivalent.
        """
        return (type(other) == type(self)
                and np.allclose(other.mat, self.mat)
                and other.nwires == self.nwires
                and other.prtcl == self.prtcl)


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

    # reverse ordering, due to convention that wire 0 corresponds to slowest varying index
    iwire   = [(nwires - 1 - iwire[m - 1 - b]) for b in range(m)]
    iwcompl = [(nwires - 1 - iwcompl[nwires - m - 1 - b]) for b in range(nwires - m)]

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
