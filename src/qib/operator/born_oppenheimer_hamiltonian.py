import numpy as np
from pyscf import gto
from qib.field import ParticleType, Field
from qib.lattice import FullyConnectedLattice
from qib.operator import AbstractOperator, FieldOperator, FieldOperatorTerm, IFOType, IFODesc


class BornOppenheimerHamiltonian(AbstractOperator):
    """
    Born-Oppenheimer Hamiltonian.
    Uses pyscf gto packages.
    Uses a fully connected lattice.
    Note that pyscf uses chemist's notation for MO integrals:
        V = sum_{i,j,k,l} v_{i,j,k,l} a+_i a_j a+_k a_l
    """
    def __init__(self, field: Field, mol):
        """
        Initialize the Hamiltonian through an already created gto molcecule.
        Save the Hamiltonian's coefficients and create a fully connected graph.
        """
        if field.particle_type != ParticleType.FERMION:
            raise ValueError(f"expecting a field with fermionic particle type, but received {field.particle_type}")
        if not isinstance(field.lattice, FullyConnectedLattice):
            raise ValueError("expecting a layered lattice when 'spin' is True")
        if field.lattice.nsites != 2*mol.nao_nr():
            raise ValueError(f"the fully connected lattice needs to have {2*mol.nao_nr()}, while {field.lattice.nsites} were given.")
        self.mol = mol
        self.h0 = self.mol.energy_nuc()
        # ACHTUNG: we care about the spin-orbital integrals
        self.h1 = np.kron(self.mol.get_hcore(), np.identity(2))
        self.h2 = self.mol.intor('int2e_spinor')
        self.field = field

    @classmethod
    def from_params(cls, atom = [], basis = "sto-3g", symmetry = None, charge = 0, spin = 0, verbose = 0):
        """
        Initialize the Hamiltonian through pyscf's gto package.
        """
        mol = gto.Mole()
        mol.build(atom = atom,
                  basis = basis,
                  symmetry = symmetry,
                  charge = charge,
                  spin = spin,
                  verbose = 0)
        return cls(mol)

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
        TODO: take into account also h0
        """
        # kinetic hopping term
        T = FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE),
                               IFODesc(self.field, IFOType.FERMI_ANNIHIL)],
                               self.h1)
        # interaction term
        V = FieldOperatorTerm([IFODesc(self.field, IFOType.FERMI_CREATE),
                               IFODesc(self.field, IFOType.FERMI_ANNIHIL),
                               IFODesc(self.field, IFOType.FERMI_CREATE),
                               IFODesc(self.field, IFOType.FERMI_ANNIHIL)],
                               self.h2)
        return FieldOperator([T, V])

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

    def fields(self):
        """
        List of fields the Hamiltonian acts on.
        """
        return [self.field]
