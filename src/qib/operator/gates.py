import abc
import numpy as np
from enum import Enum
from scipy.linalg import expm, sqrtm, block_diag
from scipy.sparse import csr_matrix
from typing import Sequence
from qib.field import Field, Particle, Qubit
from qib.operator import AbstractOperator


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
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        pass

    @abc.abstractmethod
    def inverse(self):
        """
        Return the inverse operator.
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
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        else:
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
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        else:
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
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        else:
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
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        else:
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
        else:
            return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        else:
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
        else:
            return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        else:
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
        else:
            return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        else:
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


class RotationGate(Gate):
    """
    General rotation gate; the rotation angle and axis are combined into a single vector.
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
        else:
            return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        else:
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
        else:
            return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        else:
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
        else:
            return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        else:
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
        else:
            return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        else:
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
        else:
            return []

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        if self.qubit:
            return [self.qubit.field]
        else:
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
    The control qubits have to be set separately.
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

    def inverse(self):
        """
        Return the inverse operator.
        """
        invgate = ControlledGate(self.tgate.inverse(), self.ncontrols)
        if self.control_qubits:
            invgate.set_control(self.control_qubits)
        return invgate

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
        Return the list of fields hosting the quantum particles which the gate acts on.
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
        return tprtcl + self.control_qubits

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        assert self.tgates
        for g in self.tgates:
            assert self.tgates[0].fields() == g.fields(), "fields of all target gates must match"
        return list(set(self.tgates[0].fields() + [q.field for q in self.control_qubits]))

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
        for g in self.tgates:
            if len(g.particles()) != g.num_wires:
                raise RuntimeError("unspecified target gate particle(s)")
        prtcl = self.particles()
        iwire = [_map_particle_to_wire(fields, p) for p in prtcl]
        if any([iw < 0 for iw in iwire]):
            raise RuntimeError("particle not found among fields")
        nwires = sum([f.lattice.nsites for f in fields])
        return _distribute_to_wires(nwires, iwire, csr_matrix(self.as_matrix()))


class TimeEvolutionGate(Gate):
    """
    Quantum time evolution gate, i.e., matrix exponential,
    given a Hamiltonian `h`:
    .. math:: e^{-i h t}
    """
    def __init__(self, h, t: float):
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
    def __init__(self, h, method: BlockEncodingMethod):
        self.h = h
        self.method = method
        self.auxiliary_qubits = []

    def is_hermitian(self):
        """
        Whether the gate is Hermitian.
        """
        if self.method == BlockEncodingMethod.Wx:
            return False
        elif self.method == BlockEncodingMethod.Wxi:
            return False
        elif self.method == BlockEncodingMethod.R:
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
        prtcl = []
        fields = self.h.fields()
        for f in fields:
            prtcl += [Particle(f, i) for i in range(f.lattice.nsites)]
        prtcl += self.auxiliary_qubits
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
        elif self.method == BlockEncodingMethod.Wxi:
            ginv = BlockEncodingGate(self.h, BlockEncodingMethod.Wx)
            if self.auxiliary_qubits:
                ginv.set_auxiliary_qubits(self.auxiliary_qubits)
            return ginv
        elif self.method == BlockEncodingMethod.R:
            return self
        raise NotImplementedError(f"encoding method {self.method} not supported yet")

    def fields(self):
        """
        Return the list of fields hosting the quantum particles which the gate acts on.
        """
        return list(set(self.h.fields() + [q.field for q in self.auxiliary_qubits]))

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
        elif self.method == BlockEncodingMethod.Wxi:
            return 1
        elif self.method == BlockEncodingMethod.R:
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
        """
        # assuming that `h` is Hermitian and that its spectral norm is bounded by 1
        hmat = self.h.as_matrix().toarray()
        sq1h = sqrtm(np.identity(hmat.shape[0]) - hmat @ hmat)
        if self.method == BlockEncodingMethod.Wx:
            return np.block([[hmat, 1j*sq1h], [1j*sq1h, hmat]])
        elif self.method == BlockEncodingMethod.Wxi:
            return np.block([[hmat, -1j*sq1h], [-1j*sq1h, hmat]])
        elif self.method == BlockEncodingMethod.R:
            return np.block([[hmat, sq1h], [sq1h, -hmat]])
        raise NotImplementedError(f"encoding method {self.method} not supported yet")

    def _circuit_matrix(self, fields: Sequence[Field]):
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
