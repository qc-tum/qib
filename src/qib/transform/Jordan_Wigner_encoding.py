import numpy as np
from qib.lattice import IntegerLattice, OddFaceCenteredLattice
from qib.field import ParticleType
from qib.operator import (IFOType, FieldOperator,
                          PauliString, WeightedPauliString, PauliOperator)

# Only spinless Fermi Hubbard on 2D integer lattice is implemented
# TODO: separate specific problems (es: Fermi Hubbard) that can be encoded more efficiently from a general Hamiltonian

def Jordan_Wigner_encode_field_operator(fieldop: FieldOperator):
    """
    Jordan - Wigner encoding 
    """
    fields = fieldop.fields()                                                      
    # list of all the fields. For example, Fermi-Hubbard or Born-Oppenheimer only have 1 fermionic field.
    if len(fields) != 1 or fields[0].ptype != ParticleType.FERMION:
        # currently only a single fermionic field supported
        raise NotImplementedError
    
    latt_fermi = fields[0].lattice
    # fermionic lattice (ex: Fermi-Hubbard model)
    if not isinstance(latt_fermi, IntegerLattice):
        raise RuntimeError("only integer lattices supported")
    if latt_fermi.ndim != 2:
        raise RuntimeError("only two-dimensional lattices supported")
    if any(latt_fermi.pbc):
        raise RuntimeError("only open boundary conditions supported")
    adj = latt_fermi.adjacency_matrix()
    # adjacencey matrix of the fermionic lattice
    
    latt_enc = IntegerLattice(latt_fermi.shape, pbc=False)             
    # encoded qubit lattice, same shape of fermionic lattice

    pauliop = PauliOperator()
    # string of encoded single pauli operators

    for term in fieldop.terms:
        # iterate over terms of the Hamiltonian
        # if spinless, only 2 operators for both on-site and kinetic term!
        if (len(term.opdesc) == 2
            and term.opdesc[0].otype == IFOType.FERMI_CREATE
            and term.opdesc[1].otype == IFOType.FERMI_ANNIHIL):

            if not np.issubdtype(term.coeffs.dtype, float):
                raise ValueError("only real coefficient matrices for on-site and kinetic hopping term supported")
            if not np.allclose(term.coeffs, term.coeffs.T):
                raise ValueError("only symmetric coefficient matrices for on-site and kinetic hopping term supported")
            
            # on-site term: \sum_i U_i n_{i} -----> \sum U_i 0.5*(I - Z)_{i} 
            # add identity (coeff = 0.5 for EACH on-site term) 
            id_coeff = 0

            for i in range(latt_fermi.nsites):
                z_i = PauliString.from_single_paulis(latt_enc.nsites, ('Z', i))
                pauliop.add_pauli_string(WeightedPauliString(z_i, -0.5 * term.coeffs[i, i]) )
                
                id_coeff += 0.5 * term.coeffs[i, i]                           
            
            pauliop.add_pauli_string(WeightedPauliString(PauliString.identity(latt_enc.nsites), id_coeff))

            # kinetic hopping term a_i^{\dagger} a_j + a_j^{\dagger} a_i
            for i in range(latt_fermi.nsites):
                for j in range(i + 1, latt_fermi.nsites):
                    # note: i < j
                    if term.coeffs[i, j] == 0:
                        continue
                    if adj[i, j] == 0:
                        raise ValueError("only direct neighbor hopping terms supported")
                        
                    first_kinetic_str = ['I' for k in range(i)] + ['X'] + ['Z' for k in range(j-i-1)] + ['X'] + ['I' for k in range(latt_fermi.nsites-j-1)]
                    first_kinetic_pauli = PauliString.from_string(latt_enc.nsites, first_kinetic_str)                                       
                    
                    second_kinetic_str = ['I' for k in range(i)] + ['Y'] + ['Z' for k in range(j-i-1)] + ['Y'] + ['I' for k in range(latt_fermi.nsites-j-1)]
                    second_kinetic_pauli = PauliString.from_string(latt_enc.nsites, second_kinetic_str)                                       
                    
                    pauliop.add_pauli_string(WeightedPauliString(first_kinetic_pauli,  0.5 * term.coeffs[i, j]))
                    pauliop.add_pauli_string(WeightedPauliString(second_kinetic_pauli,  0.5 * term.coeffs[i, j]))
        else:
            raise NotImplementedError

    return pauliop, latt_enc
    
    
    
   
    
    
    
    
    
    
    
    
