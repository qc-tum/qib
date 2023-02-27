import numpy as np
from qib.field import ParticleType, Field
from qib.operator import AbstractOperator, FieldOperator, FieldOperatorTerm, IFOType, IFODesc


class MolecularHamiltonian(AbstractOperator):
    """
    Molecular Hamiltonian in second quantization formulation,
    using physics convention for the interaction term:

    .. math::

        H = c + \sum_{i,j} t_{i,j} a^{\dagger}_i a_j + \\frac{1}{2} \sum_{i,j,k,\ell} v_{i,j,k,\ell} a^{\dagger}_i a^{\dagger}_j a_{\ell} a_k
    """
    def __init__(self, field: Field, c: float, tkin, vint, check=False):
        """
        Initialize the Hamiltonian by its kinetic and interaction term coefficients.
        """
        norbs = len(tkin)
        tkin = np.array(tkin, copy=False)
        vint = np.array(vint, copy=False)
        if tkin.shape != 2 * (norbs,):
            raise RuntimeError(f"tkin must have shape ({norbs}, {norbs}), instead of {tkin.shape}")
        if not np.allclose(tkin, tkin.conj().T):
            raise RuntimeError("kinetic coefficients must form a Hermitian matrix")
        if vint.shape != 4 * (norbs,):
            raise RuntimeError(f"vint must have shape ({norbs}, {norbs}, {norbs}, {norbs}), instead of {vint.shape}")
        # TODO: symmetries of vint
        if field.particle_type != ParticleType.FERMION:
            raise ValueError(f"expecting a field with fermionic particle type, but received {field.particle_type}")
        if field.lattice.nsites != norbs:
            raise ValueError(f"underlying lattice must have {norbs} sites, while {field.lattice.nsites} were given")
        self.c = c
        self.tkin = tkin
        self.vint = vint
        self.field = field
        if check:
            self._check_symm()

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
                               0.5 * self.vint)
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

    def _check_symm(self):
        """
        Checks symmetries of the 1 and 2 body integrals.
        #TODO: null elements?
        """
        if not np.allclose(self.tkin, self.tkin.conj().T):
            raise RuntimeError("Broken symmetry for 1-body operator: <i|h1|j> = <j|h1|i>*")
        if not np.allclose(self.vint, self.vint.transpose(1,0,3,2)):
            raise RuntimeError("Broken symmetry for 2-body operator: <ij|h2|lk> = <ji|h2|kl>")
        if not np.allclose(self.vint, self.vint.conj().transpose(2,3,0,1)):
            raise RuntimeError("Broken symmetry for 2-body operator: <ij|h2|lk> = <lk|h2|ij>*")
        if not np.allclose(self.vint, self.vint.conj().transpose(3,2,1,0)):
            raise RuntimeError("Broken symmetry for 2-body operator: <ij|h2|lk> = <kl|h2|ji>*")
