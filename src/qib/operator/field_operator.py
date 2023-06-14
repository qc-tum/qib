import enum
from typing import Sequence
import numpy as np
from scipy import sparse
from qib.operator import AbstractOperator
from qib.field import ParticleType, Field


class IFOType(enum.Enum):
    """
    Individual field operator type (e.g., fermionic creation operator).
    """
    BOSON_CREATE  = 1   # bosonic creation operator
    BOSON_ANNIHIL = 2   # bosonic annihilation operator
    FERMI_CREATE  = 3   # fermionic creation operator
    FERMI_ANNIHIL = 4   # fermionic annihilation operator
    MAJORANA_RE   = 5   # "real" Majorana operator
    MAJORANA_IM   = 6   # "imaginary" Majorana operator

    @staticmethod
    def adjoint(otype):
        """
        Adjoint (conjugate transpose) operator type.
        """
        if not isinstance(otype, IFOType):
            raise ValueError("incorrect argument type, expecting 'IFOType'")
        if otype == IFOType.BOSON_CREATE:
            return IFOType.BOSON_ANNIHIL
        if otype == IFOType.BOSON_ANNIHIL:
            return IFOType.BOSON_CREATE
        if otype == IFOType.FERMI_CREATE:
            return IFOType.FERMI_ANNIHIL
        if otype == IFOType.FERMI_ANNIHIL:
            return IFOType.FERMI_CREATE
        return otype

class IFODesc:
    """
    Individual field operator description: field and operator type.
    """
    def __init__(self, field: Field, otype: IFOType):
        # consistency checks
        if field.particle_type == ParticleType.BOSON:
            if otype not in (IFOType.BOSON_CREATE, IFOType.BOSON_ANNIHIL):
                raise ValueError(f"expecting bosonic operator, but received {otype}")
        elif field.particle_type == ParticleType.FERMION:
            if otype not in (IFOType.FERMI_CREATE, IFOType.FERMI_ANNIHIL):
                raise ValueError(f"expecting fermionic operator, but received {otype}")
        elif field.particle_type == ParticleType.MAJORANA:
            if otype not in (IFOType.MAJORANA_RE, IFOType.MAJORANA_IM):
                raise ValueError(f"expecting Majorana operator, but received {otype}")
        self.field = field
        self.otype = otype


class FieldOperatorTerm:
    r"""
    Field operator term in second quantization, e.g.,
    :math:`\sum_{j,k} h_{j,k} a^{\dagger}_j a_k`.

    Each summation index is associated with a field and the
    operator type (e.g., fermionic creation operator).
    """
    def __init__(self, opdesc: Sequence[IFODesc], coeffs):
        self.opdesc = tuple(opdesc)
        self.coeffs = np.asarray(coeffs)
        if self.coeffs.ndim != len(self.opdesc):
            raise ValueError("number of operator descriptions must match dimension of coefficient array")

    def is_hermitian(self):
        """
        Whether the field operator term is Hermitian.
        """
        n = len(self.opdesc)
        if not all((self.opdesc[i].field == self.opdesc[n-1-i].field) and
                   (self.opdesc[i].otype == IFOType.adjoint(self.opdesc[n-1-i].otype))
                   for i in range(n)):
            return False
        return np.allclose(self.coeffs, self.coeffs.conj().T)

    def fields(self):
        """
        List of all fields appearing in the term.
        """
        f_list = []
        for desc in self.opdesc:
            if desc.field not in f_list:
                f_list.append(desc.field)
        return f_list

    def __matmul__(self, other):
        """
        Logical product of two field operator terms.
        """
        if not isinstance(other, FieldOperatorTerm):
            raise ValueError("expecting another field operator term for multiplication")
        coeffs = np.kron(self.coeffs.reshape(-1),
                        other.coeffs.reshape(-1)).reshape(self.coeffs.shape + other.coeffs.shape)
        return FieldOperatorTerm(self.opdesc + other.opdesc, coeffs)


class FieldOperator(AbstractOperator):
    """
    Field operator in second quantized form.
    """
    def __init__(self, terms: Sequence[FieldOperatorTerm]=None):
        if terms is None:
            self.terms = []
        else:
            self.terms = list(terms)

    def fields(self):
        """
        List of all fields appearing in the operator.
        """
        f_list = []
        for term in self.terms:
            for f in term.fields():
                if f not in f_list:
                    f_list.append(f)
        return f_list

    def is_unitary(self):
        """
        Whether the operator is unitary.
        """
        # might be unitary in special cases, but difficult to check,
        # so returning False here for simplicity
        return False

    def is_hermitian(self):
        """
        Whether the operator is Hermitian.
        """
        if all(term.is_hermitian() for term in self.terms):
            return True
        # sum of two terms can be Hermitian, although the individual
        # terms are not, like the superconducting pairing term
        raise NotImplementedError

    def __matmul__(self, other):
        """
        Logical product of two field operators.
        """
        if not isinstance(other, FieldOperator):
            raise ValueError("expecting another field operator for multiplication")
        # take all pairwise products
        return FieldOperator([t1 @ t2 for t1 in self.terms for t2 in other.terms])

    def as_matrix(self):
        """
        Generate the (sparse) matrix representation of the operator.
        """
        fields = self.fields()
        if len(fields) != 1 or fields[0].ptype != ParticleType.FERMION:
            # currently only a single fermionic field supported
            raise NotImplementedError
        # number of lattice sites
        L = fields[0].lattice.nsites
        # assemble fermionic creation operators based on Jordan-Wigner transformation
        I = sparse.identity(2)
        Z = sparse.csr_matrix([[ 1.,  0.], [ 0., -1.]])
        U = sparse.csr_matrix([[ 0.,  0.], [ 1.,  0.]])
        clist = []
        for i in range(L):
            c = sparse.identity(1)
            for j in range(L):
                if j < i:
                    c = sparse.kron(c, I)
                elif j == i:
                    c = sparse.kron(c, U)
                else:
                    c = sparse.kron(c, Z)
            clist.append(c)
        # corresponding annihilation operators
        alist = [c.conj().T for c in clist]
        # assemble overall field operator
        op = sparse.csr_matrix((2**L, 2**L))
        for term in self.terms:
            it = np.nditer(term.coeffs, flags=["multi_index"])
            for coeff in it:
                if coeff == 0:
                    continue
                fstring = sparse.identity(2**L)
                for i, j in enumerate(it.multi_index):
                    if term.opdesc[i].otype == IFOType.FERMI_CREATE:
                        fstring = fstring @ clist[j]
                    elif term.opdesc[i].otype == IFOType.FERMI_ANNIHIL:
                        fstring = fstring @ alist[j]
                    else:
                        raise RuntimeError(f"expecting fermionic operator, but received {term.opdesc[i].otype}")
                op += coeff * fstring
        return op
