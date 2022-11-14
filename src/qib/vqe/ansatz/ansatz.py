import numpy as np
from scipy import sparse
from scipy.linalg import expm
from typing import Sequence
from qib.operator import (AbstractOperator, IFOType, IFODesc, FieldOperator, FieldOperatorTerm)
from qib.transform import jordan_wigner_encode_field_operator
from qib.lattice import LayeredLattice


class Ansatz(AbstractOperator):
    """
    Parent class for VQE ansatze.
    """


class qUCC(Ansatz):
    """
    Quantum Unitary Coupled Cluster ansatz (generalized).
    Can be with single or double excitations.
    Can work with spin or spinless fermions (should be coherent with the Hamiltonian!).
    When spin_symmetry = True the amount of parameters is reduced by considering symmetry between spin up and down.
    """
    def __init__(self, field: FieldOperator, excitations: str="s", embedding: str="jordan_wigner", spin=False):
        if spin and not isinstance(FieldOperator, LayeredLattice):
            raise ValueError(f"When 'spin=True', a LayeredLattice is needed.")
        self.field = field
        if not set(excitations).issubset({"s","d","sd"}):
            raise ValueError(f"The only options for excitations are single 's', double 'd' or single and double 'sd', while {excitations} was given.")
        self.excitations = excitations
        if not embedding == "jordan_wigner":
            raise NotImplementedError(f"Only 'jordan_wigner' encoding is implemented.")
        self.embedding = embedding
        if spin:
            raise NotImplementedError(f"spin==True case not implemented yet")
        self.spin = spin

    # TODO: make it more general (different embeddings)
    # TODO: add spin option and only right excitations
    @property
    def nqubits(self):
        """
        Number of qubits is equal to number of fermionic sites.
        This is true only for Jordan Wigner!
        """
        return self.field.lattice.nsites

    @property
    def num_parameters(self):
        """
        Number of variational parameters.
        """
        if self.excitations == "s":
            return self.nqubits**2
        elif self.excitations == "d":
            return self.nqubits**4
        else:
            return self.nqubits**2+self.nqubits**4

    def is_unitary(self):
        """
        qUCC is always unitary.
        """
        return True

    def is_hermitian(self):
        """
        Generally speaking, qUCC is not hermitian.
        """
        return False

    # TODO: distinguish between spin
    def as_matrix(self, params: Sequence[float]):
        """
        Generate the (sparse) matrix representation of the ansatz.
        Here we used the generalized version of UCC (all excitation terms allowed among the same spin set).
        The single and double excitations terms are trotterized and Jordan Wigner is separately applied to them.
        """
        params = np.array(params)
        if self.excitations == "s":
            if not len(params) == self.nqubits**2:
                raise ValueError(f"For {self.excitations} excitations, {self.nqubits**2} 'params' are needed while {len(params)} were given.")
            params = np.reshape(params, (self.nqubits, self.nqubits))
            T = FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE),IFODesc(self.field, IFOType.FERMI_ANNIHIL)],params)
            T_pauli = jordan_wigner_encode_field_operator(FieldOperator([T]))
            T_mat = T_pauli.as_matrix().toarray()
            return sparse.csr_matrix(expm(T_mat - T_mat.conjugate().T))

        elif self.excitations == "d":
            if not len(params) == self.nqubits**4:
                raise ValueError(f"For {self.excitations} excitations, {self.nqubits**4} 'params' are needed while {len(params)} were given.")
            params = np.reshape(params, (self.nqubits, self.nqubits, self.nqubits, self.nqubits))
            T = FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE),IFODesc(self.field, IFOType.FERMI_CREATE),
                                   IFODesc(self.field, IFOType.FERMI_ANNIHIL),IFODesc(self.field, IFOType.FERMI_ANNIHIL)],params)
            T_pauli = jordan_wigner_encode_field_operator(FieldOperator([T]))
            T_mat = T_pauli.as_matrix().toarray()
            return sparse.csr_matrix(expm(T_mat - T_mat.conjugate().T))

        elif self.excitations == "sd":
            if not len(params) == self.nqubits**2 + self.nqubits**4:
                raise ValueError(f"For {self.excitations} excitations, {self.nqubits**2 + self.nqubits**4} 'params' are needed while {len(params)} were given.")
            params = [params[:self.nqubits**2]] + [params[self.nqubits**2:]]
            params[0] = np.reshape(params[0], (self.nqubits, self.nqubits))
            params[1] = np.reshape(params[1], (self.nqubits, self.nqubits, self.nqubits, self.nqubits))
            T = [FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE),IFODesc(self.field, IFOType.FERMI_ANNIHIL)],params[0]),
                 FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE),IFODesc(self.field, IFOType.FERMI_CREATE),
                                     IFODesc(self.field, IFOType.FERMI_ANNIHIL),IFODesc(self.field, IFOType.FERMI_ANNIHIL)],params[1])]
            U = []
            for i in range(2):
                T_pauli = jordan_wigner_encode_field_operator(FieldOperator([T[i]]))
                T_mat = T_pauli.as_matrix().toarray()
                U.append(expm(T_mat - T_mat.conjugate().T))
            return sparse.csr_matrix(U[0]@U[1])
