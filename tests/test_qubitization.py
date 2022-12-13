import numpy as np
import unittest
import sys
sys.path.append('../src')
import qib


class TestQubitization(unittest.TestCase):

    def test_eigenvalue_transformation(self):
        """
        Test the eigenvalue transformation
        """
        # construct a simple Hamiltonian
        L = 2
        latt = qib.lattice.IntegerLattice((L,), pbc=True)
        field1 = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
        #np.random.seed(0)
        H = qib.operator.HeisenbergHamiltonian(field1, np.random.standard_normal(size=3),
                                                       np.random.standard_normal(size=3))
        # rescale parameters (effectively rescales overall Hamiltonian)
        scale = 1.25 * np.linalg.norm(H.as_matrix().toarray(), ord=2)
        H.J /= scale
        H.h /= scale
        # auxiliary qubits
        # Note: if I choose a bigger lattice the circuit will hold all of it (comparison fails)
        field2 = qib.field.Field(qib.field.ParticleType.QUBIT,
                                 qib.lattice.IntegerLattice((2,), pbc=False))
        q_enc = qib.field.Qubit(field2, 0)
        q_anc = qib.field.Qubit(field2, 1)
        for method in qib.operator.BlockEncodingMethod:
            block = qib.BlockEncodingGate(H, method)
            block.set_auxiliary_qubits(q_enc)
            processing = qib.ProjectorControlledPhaseShift(q_enc, q_anc)
            theta = [0.]
            eigen_transform = qib.algorithms.qubitization.EigenvalueTransformation(h=H, 
                                                                                   method=method, 
                                                                                   q_enc=q_enc,
                                                                                   q_anc=q_anc, 
                                                                                   projector=[1,0], 
                                                                                   theta_seq=theta)
            eigen_transform_gate = qib.EigenvalueTransformationGate(block, processing, theta)
            # if theta == 0 I only have block unitary
            self.assertTrue(np.allclose(np.kron(np.identity(2**1), block.as_matrix()), eigen_transform.as_circuit().as_matrix([field1,field2]).toarray()))
            # if theta == [0,0] I have identity
            theta = [0,0]
            eigen_transform.set_theta_seq(theta)
            self.assertTrue(np.allclose(np.identity(2**(latt.nsites+field2.lattice.nsites)), eigen_transform.as_circuit().as_matrix([field1,field2]).toarray()))
            # random theta
            #theta = [np.random.uniform(0, 2*np.pi) for i in range(10)]
            theta = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(1,10):
                eigen_transform.set_theta_seq(theta[:i])
                eigen_transform_gate.set_theta_seq(theta[:i])
                mat_class = eigen_transform.as_matrix()
                mat_gate = eigen_transform_gate.as_matrix()
                circ_class = eigen_transform.as_circuit().as_matrix([field1,field2]).toarray()
                circ_gate = eigen_transform_gate._circuit_matrix([field1, field2]).toarray()
                #print(i, theta[:i])
                #print(eigen_transform.as_circuit().as_matrix([field1,field2]))
                #print(eigen_transform_gate._circuit_matrix([field1, field2]))
                self.assertTrue(np.allclose(mat_class, mat_gate))
                self.assertTrue(np.allclose(mat_class, circ_gate))
                '''
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            for l in range(4):
                                perm = [i,j,k,l]
                                if len(perm) == len(list(set(perm))):
                                    if np.allclose(qib.util.permute_gate_wires(circ_class, perm), mat_class):
                                        print(perm)
                '''
                
                self.assertTrue(np.allclose(mat_class, circ_class))

            '''
            self.assertEqual(gate.num_wires, L + 1)
            self.assertTrue(gate.encoded_operator() is H)
            gmat = gate.as_matrix()
            self.assertTrue(np.allclose(gmat @ gmat.conj().T,
                                        np.identity(2**gate.num_wires)))
            self.assertTrue(np.allclose(gmat @ gate.inverse().as_matrix(),
                                        np.identity(2**gate.num_wires)))
            gate.set_auxiliary_qubits([q])
            self.assertTrue(gate.fields() == [field1, field2] or gate.fields() == [field2, field1])
            # principal quantum state
            ψp = qib.util.crandn(2**L)
            ψp /= np.linalg.norm(ψp)
            # quantum state on auxiliary register
            ψa = np.kron(np.kron(qib.util.crandn(4), [1, 0]), qib.util.crandn(2))
            ψa /= np.linalg.norm(ψa)
            # overall quantum state
            ψ = np.kron(ψa, ψp)
            # projection |0><0| acting on auxiliary qubit
            Pa = sparse.kron(sparse.kron(sparse.identity(4), sparse.diags([1., 0.])), sparse.identity(2**(L + 1)))
            # text block-encoding of Hamiltonian
            self.assertTrue(np.allclose(Pa @ (gate._circuit_matrix([field1, field2]) @ ψ),
                                        np.kron(ψa, H.as_matrix() @ ψp)))
            '''

    '''
    def test_handmade(self):
        """
        Test the eigenvalue transformation
        """
        # construct a simple Hamiltonian
        L = 2
        latt = qib.lattice.IntegerLattice((L,), pbc=True)
        field1 = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
        #np.random.seed(0)
        H = qib.operator.HeisenbergHamiltonian(field1, np.random.standard_normal(size=3),
                                                       np.random.standard_normal(size=3))
        # rescale parameters (effectively rescales overall Hamiltonian)
        scale = 1.25 * np.linalg.norm(H.as_matrix().toarray(), ord=2)
        H.J /= scale
        H.h /= scale
        # auxiliary qubits
        # Note: if I choose a bigger lattice the circuit will hold all of it (comparison fails)
        field2 = qib.field.Field(qib.field.ParticleType.QUBIT,
                                 qib.lattice.IntegerLattice((2,), pbc=False))
        q_enc = qib.field.Qubit(field2, 0)
        q_anc = qib.field.Qubit(field2, 1)
        for method in qib.operator.BlockEncodingMethod:
            block = qib.BlockEncodingGate(H, method)
            block.set_auxiliary_qubits(q_enc)
            processing = qib.ProjectorControlledPhaseShift(q_enc, q_anc)
            processing_1 = qib.ProjectorControlledPhaseShift(q_enc, q_anc)
            theta = [1.2, 0.5]
            eigen_transform = qib.algorithms.qubitization.EigenvalueTransformation(h=H, 
                                                                                   method=method, 
                                                                                   q_enc=q_enc,
                                                                                   q_anc=q_anc, 
                                                                                   projector=[1,0], 
                                                                                   theta_seq=theta)
            

            matrix = np.identity(2**eigen_transform.num_wires)
            num_particles = block.num_wires - block.num_aux_qubits
            id_for_projector = np.identity(2**num_particles)
            id_for_unitary = np.identity(2**len(processing.auxiliary_qubits))
            U_inv_matrix = block.inverse().as_matrix()
            U_matrix = block.as_matrix()
            circuit = qib.Circuit()
            if(len(theta)%2==0):
                dim = len(theta)//2
                start = 0
            else:
                dim = (len(theta)-1)//2
                processing.set_theta(theta[0])
                matrix = matrix @ np.kron(processing.as_matrix(), id_for_projector) \
                                @ np.kron(id_for_unitary, U_matrix)
                circuit.prepend_gate(processing)
                circuit.prepend_gate(block)

                # OK!
                print(0, theta[0], np.allclose(circuit.as_matrix([field1, field2]).toarray(), matrix))
                start = 1
            for i in range(start, dim):
                matrix = np.identity(2**eigen_transform.num_wires)
                circuit = qib.Circuit()
                processing.set_theta(theta[2*i-start])
                matrix = matrix @ np.kron(processing.as_matrix(), id_for_projector) \
                                @ np.kron(id_for_unitary, U_inv_matrix) 
                circuit.prepend_gate(processing)
                circuit.prepend_gate(block.inverse())
                # OK!
                print(i, theta[2*i-start], np.allclose(circuit.as_matrix([field1, field2]).toarray(), matrix))

                #matrix = np.identity(2**eigen_transform.num_wires)
                #circuit = qib.Circuit()
                processing.set_theta(theta[2*i+1-start])
                matrix = matrix @ np.kron(processing.as_matrix(), id_for_projector) \
                                @ np.kron(id_for_unitary, U_matrix)
                circuit.prepend_gate(processing)
                circuit.prepend_gate(block)
                print(i, theta[2*i+1-start], np.allclose(circuit.as_matrix([field1, field2]).toarray(), matrix))


                matrix = np.identity(2**eigen_transform.num_wires)
                circuit = qib.Circuit()
                processing.set_theta(theta[2*i-start])
                processing_1.set_theta(theta[2*i+1-start])
                matrix = matrix @ np.kron(processing.as_matrix(), id_for_projector) \
                                @ np.kron(id_for_unitary, U_inv_matrix) \
                                @ np.kron(processing_1.as_matrix(), id_for_projector) \
                                @ np.kron(id_for_unitary, U_matrix)
                circuit.prepend_gate(processing)
                circuit.prepend_gate(block.inverse())
                circuit.prepend_gate(processing_1)
                circuit.prepend_gate(block)
                print(i, np.allclose(circuit.as_matrix([field1, field2]).toarray(), matrix))
            self.assertTrue(np.allclose(matrix, eigen_transform.as_circuit().as_matrix([field1,field2]).toarray()))
    '''

if __name__ == "__main__":
    unittest.main()
