import numpy as np
from qib.field import ParticleType
from qib.operator import (IFOType, FieldOperator,
                          PauliString, WeightedPauliString, PauliOperator)


def jordan_wigner_encode_field_operator(fieldop: FieldOperator) -> PauliOperator:
    """
    Jordan-Wigner encode a fermionic field operator.
    """
    fields = fieldop.fields()
    if len(fields) != 1 or fields[0].ptype != ParticleType.FERMION:
        # currently only a single fermionic field supported
        raise NotImplementedError("only a single fermionic field supported")

    # number of lattice sites
    L = fields[0].lattice.nsites
    # represent fermionic operators as Pauli strings based on Jordan-Wigner transformation
    clist = []
    alist = []
    for i in range(L):
        za = i*[0] + [0] + (L-i-1)*[1]
        zb = i*[0] + [1] + (L-i-1)*[1]
        x  = i*[0] + [1] + (L-i-1)*[0]
        # require two Pauli strings per fermionic operator
        clist.append([PauliString(za, x, 0), PauliString(zb, x, 1)])
        alist.append([PauliString(za, x, 0), PauliString(zb, x, 3)])

    # assemble overall operator
    pauliop = PauliOperator()
    for term in fieldop.terms:
        it = np.nditer(term.coeffs, flags=["multi_index"])
        for coeff in it:
            if coeff == 0:
                continue
            pstrings = [PauliString.identity(L)]
            for i, j in enumerate(it.multi_index):
                if term.opdesc[i].otype == IFOType.FERMI_CREATE:
                    pstrings = (  [ps @ clist[j][0] for ps in pstrings]
                                + [ps @ clist[j][1] for ps in pstrings])
                elif term.opdesc[i].otype == IFOType.FERMI_ANNIHIL:
                    pstrings = (  [ps @ alist[j][0] for ps in pstrings]
                                + [ps @ alist[j][1] for ps in pstrings])
                else:
                    raise RuntimeError(f"expecting fermionic operator, but received {term.opdesc[i].otype}")
            # scaling factors 1/2 from representation of each fermionic operator as two Pauli strings
            weight = 0.5 ** len(term.opdesc) * coeff
            for ps in pstrings:
                # include overall sign factor in weight coefficient;
                # factoring out phase (instead of sign only)
                # does not seem to be advantageous
                sign = ps.refactor_sign()
                pauliop.add_pauli_string(WeightedPauliString(ps, sign * weight))
    pauliop.remove_zero_weight_strings(tol=1e-14)

    return pauliop
