import numpy as np
from qib.lattice import IntegerLattice, OddFaceCenteredLattice
from qib.operator import (ParticleType, IFOType, FieldOperator,
                          PauliString, WeightedPauliString, PauliOperator)


def compact_encode_field_operator(fieldop: FieldOperator):
    """
    "Compact" encode a fermionic field operator on a square lattice
    into a qubit Hamiltonian.

    Reference:
        Charles Derby, Joel Klassen, Johannes Bausch, Toby Cubitt
        Compact fermion to qubit mappings
        Phys. Rev. B 104, 035118 (2021)
    """
    fields = fieldop.fields()
    if len(fields) != 1 or fields[0].ptype != ParticleType.FERMION:
        # currently only a single fermionic field supported
        raise NotImplementedError
    latt_fermi = fields[0].lattice
    if not isinstance(latt_fermi, IntegerLattice):
        raise RuntimeError("only integer lattices supported")
    if latt_fermi.ndim != 2:
        raise RuntimeError("only two-dimensional lattices supported")
    if any(latt_fermi.pbc):
        raise RuntimeError("only open boundary conditions supported")
    adj = latt_fermi.adjacency_matrix()

    latt_enc = OddFaceCenteredLattice(latt_fermi.shape, pbc=False)

    pauliop = PauliOperator()

    for term in fieldop.terms:

        if (len(term.opdesc) == 2
            and term.opdesc[0].otype == IFOType.FERMI_CREATE
            and term.opdesc[1].otype == IFOType.FERMI_ANNIHIL):

            if not np.issubdtype(term.coeffs.dtype, float):
                raise ValueError("only real coefficient matrices for on-site and kinetic hopping term supported")
            if not np.allclose(term.coeffs, term.coeffs.T):
                raise ValueError("only symmetric coefficient matrices for on-site and kinetic hopping term supported")

            # on-site term
            id_coeff = 0
            for i in range(latt_fermi.nsites):
                ic = latt_fermi.index_to_coord(i)
                Vi = _encode_vertex_operator(latt_enc, ic)
                pauliop.add_pauli_string(WeightedPauliString(Vi, -0.5 * term.coeffs[i, i]))
                id_coeff += 0.5 * term.coeffs[i, i]
            # add identity
            pauliop.add_pauli_string(WeightedPauliString(PauliString.identity(latt_enc.nsites), id_coeff))

            # kinetic hopping term a_i^{\dagger} a_j + a_j^{\dagger} a_i
            for i in range(latt_fermi.nsites):
                for j in range(i + 1, latt_fermi.nsites):
                    # note: i < j
                    if term.coeffs[i, j] == 0:
                        continue
                    if adj[i, j] == 0:
                        raise ValueError("only direct neighbor hopping terms supported")
                    ic = latt_fermi.index_to_coord(i)
                    jc = latt_fermi.index_to_coord(j)
                    E = _encode_edge_operator(latt_enc, ic, jc)
                    Vi = _encode_vertex_operator(latt_enc, ic)
                    Vj = _encode_vertex_operator(latt_enc, jc)
                    pauliop.add_pauli_string(WeightedPauliString(E @ Vj,  0.5j * term.coeffs[i, j]))
                    pauliop.add_pauli_string(WeightedPauliString(E @ Vi, -0.5j * term.coeffs[i, j]))
        else:
            raise NotImplementedError

    return pauliop, latt_enc


def _encode_vertex_operator(latt: OddFaceCenteredLattice, j):
    """
    Construct the vertex operator V_j, given the coordinate j = (jx, jy).
    """
    j_idx = latt.coord_to_index(j)
    return PauliString.from_single_paulis(latt.nsites, ('Z', j_idx))


def _encode_edge_operator(latt: OddFaceCenteredLattice, i, j):
    """
    Construct the edge operator E_{ij},
    given vertex coordinates i = (ix, iy) and j = (jx, jy).
    """
    # unpack vertex coordinates
    ix, iy = i
    jx, jy = j
    # ensure that (i, j) are nearest neighbors
    assert (ix == jx and abs(iy - jy) == 1) or (iy == jy and abs(ix - jx) == 1), "not a nearest neighbor edge"
    i_idx = latt.coord_to_index(i)
    j_idx = latt.coord_to_index(j)
    f_idx = latt.edge_to_odd_face_index(i, j)
    if ix == jx:    # horizontal edge
        if (ix % 2 == 0 and jy < iy) or (ix % 2 == 1 and iy < jy):
            # conforming horizontal orientation
            E = PauliString.from_single_paulis(latt.nsites, ('X', i_idx), ('Y', j_idx))
        else:
            # setting q = 2 corresponds to global (-1) factor
            E = PauliString.from_single_paulis(latt.nsites, ('X', j_idx), ('Y', i_idx), q=2)
        if f_idx != -1:
            E.set_pauli('Y', f_idx)
    else:       # vertical edge
        assert iy == jy
        if iy % 2 == 0: # iy even
            if jx < ix:     # conforming up orientation
                # setting q = 2 corresponds to global (-1) factor
                E = PauliString.from_single_paulis(latt.nsites, ('X', i_idx), ('Y', j_idx), q=2)
            else:           # non-conforming down orientation
                E = PauliString.from_single_paulis(latt.nsites, ('X', j_idx), ('Y', i_idx))
        else:   # iy odd
            if ix < jx:     # conforming down orientation
                E = PauliString.from_single_paulis(latt.nsites, ('X', i_idx), ('Y', j_idx))
            else:           # non-conforming up orientation
                # setting q = 2 corresponds to global (-1) factor
                E = PauliString.from_single_paulis(latt.nsites, ('X', j_idx), ('Y', i_idx), q=2)
        if f_idx != -1:
            E.set_pauli('X', f_idx)
    return E
