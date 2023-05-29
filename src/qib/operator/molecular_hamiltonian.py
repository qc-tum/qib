from enum import Flag, auto
import numpy as np
from qib.field import ParticleType, Field
from qib.operator import AbstractOperator, FieldOperator, FieldOperatorTerm, IFOType, IFODesc


class MolecularHamiltonianSymmetry(Flag):
    """
    Symmetries of a molecular Hamiltonian.
    """
    HERMITIAN = auto()  # Hermitian
    VARCHANGE = auto()  # symmetric w.r.t. integration variable interchange in two-body coefficients


class MolecularHamiltonian(AbstractOperator):
    """
    Molecular Hamiltonian in second quantization formulation,
    using physicists' convention for the interaction term (note ordering of k and \ell):

    .. math::

        H = c + \sum_{i,j} t_{i,j} a^{\dagger}_i a_j + \\frac{1}{2} \sum_{i,j,k,\ell} v_{i,j,k,\ell} a^{\dagger}_i a^{\dagger}_j a_{\ell} a_k
    """
    def __init__(self, field: Field, c: float, tkin, vint, symm: MolecularHamiltonianSymmetry):
        """
        Initialize the Hamiltonian by its kinetic and interaction term coefficients.
        """
        norbs = len(tkin)
        tkin = np.asarray(tkin)
        vint = np.asarray(vint)
        if tkin.shape != 2 * (norbs,):
            raise ValueError(f"tkin must have shape ({norbs}, {norbs}), instead of {tkin.shape}")
        if vint.shape != 4 * (norbs,):
            raise ValueError(f"vint must have shape {4 * (norbs,)}, instead of {vint.shape}")
        if field.particle_type != ParticleType.FERMION:
            raise ValueError(f"expecting a field with fermionic particle type, but received {field.particle_type}")
        if field.lattice.nsites != norbs:
            raise ValueError(f"underlying lattice must have {norbs} sites, while {field.lattice.nsites} were given")
        # check symmetries of the 1- and 2-body term coefficients
        if MolecularHamiltonianSymmetry.HERMITIAN in symm:
            if not isinstance(c, (int, float)):
                raise ValueError(f"constant coefficient must be a real number, received {c}")
            if not np.allclose(tkin, tkin.conj().T):
                raise ValueError("kinetic coefficients must be Hermitian")
            if not np.allclose(vint, vint.conj().transpose(2, 3, 0, 1)):
                raise ValueError("interaction operator coefficients not Hermitian, expecting <ij|kl> = <kl|ij>*")
        if MolecularHamiltonianSymmetry.VARCHANGE in symm:
            if not np.allclose(vint, vint.transpose(1, 0, 3, 2)):
                raise ValueError("interaction operator coefficients not symmetric w.r.t. variable interchange, expecting <ij|kl> = <ji|lk>")
        self.field = field
        self.c = c
        self.tkin = tkin
        self.vint = vint
        self.symm = symm

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
        return MolecularHamiltonianSymmetry.HERMITIAN in self.symm

    def as_field_operator(self):
        """
        Represent the Hamiltonian as FieldOperator.
        """
        # constant term
        C = FieldOperatorTerm([], np.array(self.c))  # NumPy array has degree 0
        # kinetic hopping term
        T = FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE),
                               IFODesc(self.field, IFOType.FERMI_ANNIHIL)],
                               self.tkin)
        # interaction term
        V = FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE),
                               IFODesc(self.field, IFOType.FERMI_CREATE),
                               IFODesc(self.field, IFOType.FERMI_ANNIHIL),
                               IFODesc(self.field, IFOType.FERMI_ANNIHIL)],
                               0.5 * self.vint.transpose((0, 1, 3, 2)))
        return FieldOperator([C, T, V])

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

    @property
    def num_orbitals(self) -> int:
        """
        Number of orbitals (same as number of underlying lattice sites).
        """
        return self.nsites

    def fields(self):
        """
        List of fields the Hamiltonian acts on.
        """
        return [self.field]
