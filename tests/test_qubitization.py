import numpy as np
from scipy import sparse
import unittest
import qib


# class TestQubitization(unittest.TestCase):

#     def test_eigenvalue_transformation(self):
#         """
#         Test the eigenvalue transformation
#         """
#         # construct a simple Hamiltonian
#         L = 2
#         latt = qib.lattice.IntegerLattice((L,), pbc=True)
#         field1 = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
#         H = qib.operator.HeisenbergHamiltonian(field1, np.random.standard_normal(size=3),
#                                                        np.random.standard_normal(size=3))
#         # rescale parameters (effectively rescales overall Hamiltonian)
#         scale = 1.25 * np.linalg.norm(H.as_matrix().toarray(), ord=2)
#         H.J /= scale
#         H.h /= scale
#         # auxiliary qubits
#         # Note: if I choose a bigger lattice the circuit will hold all of it (comparison fails)
#         field2 = qib.field.Field(qib.field.ParticleType.QUBIT,
#                                  qib.lattice.IntegerLattice((2,), pbc=False))
#         q_enc = qib.field.Qubit(field2, 0)
#         q_anc = qib.field.Qubit(field2, 1)
#         for method in qib.operator.BlockEncodingMethod:
#             gate = qib.BlockEncodingGate(H, method)
#             gate.set_auxiliary_qubits(q_enc)
#             processing = qib.ProjectorControlledPhaseShift(q_enc, q_anc)
#             #theta = np.random.uniform(0, 2*np.pi)
#             theta = np.pi/4.
#             eigen_transform = qib.algorithms.qubitization.EigenvalueTransformation(h=H,
#                                                                                    method=method,
#                                                                                    q_enc=q_enc,
#                                                                                    q_anc=q_anc,
#                                                                                    projector=[1,0],
#                                                                                    theta_seq=[theta])
#             eigen_transform_gate = qib.EigenvalueTransformationGate(gate, processing, [theta])
#             self.assertTrue(np.allclose(eigen_transform.as_matrix(), eigen_transform_gate.as_matrix()))
#             self.assertTrue(np.allclose(eigen_transform.as_matrix(), eigen_transform_gate._circuit_matrix([field1, field2]).toarray()))
#             # ??????????????????????
#             #print(eigen_transform.as_matrix())
#             #print(eigen_transform.as_circuit().as_matrix([field1, field2]))
#             #print(eigen_transform_gate._circuit_matrix([field1, field2]))
#             self.assertTrue(np.allclose(eigen_transform.as_circuit().as_matrix([field1, field2]).toarray(), eigen_transform_gate._circuit_matrix([field1, field2]).toarray()))
#             mat = eigen_transform.as_matrix()
#             circ = eigen_transform.as_circuit().as_matrix([field1, field2]).toarray()
#             self.assertTrue(np.allclose(mat, circ))

#             '''
#             self.assertEqual(gate.num_wires, L + 1)
#             self.assertTrue(gate.encoded_operator() is H)
#             gmat = gate.as_matrix()
#             self.assertTrue(np.allclose(gmat @ gmat.conj().T,
#                                         np.identity(2**gate.num_wires)))
#             self.assertTrue(np.allclose(gmat @ gate.inverse().as_matrix(),
#                                         np.identity(2**gate.num_wires)))
#             gate.set_auxiliary_qubits([q])
#             self.assertTrue(gate.fields() == [field1, field2] or gate.fields() == [field2, field1])
#             # principal quantum state
#             ψp = qib.util.crandn(2**L)
#             ψp /= np.linalg.norm(ψp)
#             # quantum state on auxiliary register
#             ψa = np.kron(np.kron(qib.util.crandn(4), [1, 0]), qib.util.crandn(2))
#             ψa /= np.linalg.norm(ψa)
#             # overall quantum state
#             ψ = np.kron(ψa, ψp)
#             # projection |0><0| acting on auxiliary qubit
#             Pa = sparse.kron(sparse.kron(sparse.identity(4), sparse.diags([1., 0.])), sparse.identity(2**(L + 1)))
#             # text block-encoding of Hamiltonian
#             self.assertTrue(np.allclose(Pa @ (gate._circuit_matrix([field1, field2]) @ ψ),
#                                         np.kron(ψa, H.as_matrix() @ ψp)))
#             '''

# def permute_gate_wires(u: np.ndarray, perm):
#     """
#     Transpose (permute) the wires of a quantum gate stored as NumPy array.
#     """
#     nwires = len(perm)
#     assert u.shape == (2**nwires, 2**nwires)
#     perm = list(perm)
#     u = np.reshape(u, (2*nwires) * (2,))
#     u = np.transpose(u, perm + [nwires + p for p in perm])
#     u = np.reshape(u, (2**nwires, 2**nwires))
#     return u


# if __name__ == "__main__":
#     unittest.main()
