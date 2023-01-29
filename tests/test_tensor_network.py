import numpy as np
import unittest
import qib


class TestTensorNetwork(unittest.TestCase):

    def test_tensor_network(self):
        """
        Test tensor network functionality.
        """
        # random number generator
        rng = np.random.default_rng()

        # first network
        stn1 = qib.tensor_network.SymbolicTensorNetwork()
        stn1.add_tensor(qib.tensor_network.SymbolicTensor( 3, ( 3,  5,  6,  2),         ( 3, -7, -3, 14),         "a"))
        stn1.add_tensor(qib.tensor_network.SymbolicTensor( 1, ( 7,  4,  2,  5),         ( 6, 17, 14, -7),         "b"))
        stn1.add_tensor(qib.tensor_network.SymbolicTensor( 7, ( 7,  9,  3, 17, 17),     ( 6, 13,  3,  5,  5),     "c06"))
        stn1.add_tensor(qib.tensor_network.SymbolicTensor( 0, (13,  7,  3),             ( 2,  0,  3),             "dd"))
        stn1.add_tensor(qib.tensor_network.SymbolicTensor( 5, (13, 11,  6),             ( 2, 11, -3),             "e"))
        stn1.add_tensor(qib.tensor_network.SymbolicTensor(-1, ( 7,  9, 11, 17,  4,  7), ( 0, 13, 11,  5, 17,  0), None))  # virtual tensor for open axes
        stn1.generate_bonds()
        self.assertTrue(stn1.is_consistent())
        ntens1 = stn1.num_tensors
        nbond1 = stn1.num_bonds
        stn1.rename_tensor(7, 8)
        self.assertTrue(stn1.is_consistent())
        stn1.rename_bond(14, 10)
        self.assertTrue(stn1.is_consistent())
        # generate random entries for the tensors in the network
        data_dict1 = { tensor.dataref: rng.normal(size=tensor.shape) for tensor in stn1.tensors.values() if tensor.tid != -1 }
        net1 = qib.tensor_network.TensorNetwork(stn1, data_dict1)
        self.assertTrue(net1.is_consistent())

        # network contraction by a single call of np.einsum
        net1_einsum_contracted, axes_map_einsum = net1.contract_einsum()
        self.assertEqual(tuple(net1_einsum_contracted.shape[i] for i in axes_map_einsum), net1.shape)

        # binary tree contraction of network
        net1_tree_contracted, axes_map_tree, tree = net1.contract_tree([[[3, 1], [0, 8]], 5])
        self.assertEqual(tuple(net1_tree_contracted.shape[i] for i in axes_map_tree), net1.shape)
        # these need not agree in general
        self.assertEqual(axes_map_einsum, axes_map_tree)
        # compare einsum with tree contraction
        self.assertTrue(np.allclose(net1_einsum_contracted, net1_tree_contracted))

        # probe axes permutations
        node = tree.children[0].children[1].children[1]
        perm = rng.permutation(node.ndim)
        node.permute_axes(perm)
        tensor_dict = { tensor.tid: data_dict1[tensor.dataref] for tensor in stn1.tensors.values() if tensor.tid != -1 }
        tensor_dict[node.tid] = tensor_dict[node.tid].transpose(perm)
        net1_tree_contracted2 = qib.tensor_network.contraction_tree.perform_tree_contraction(tree, tensor_dict)
        self.assertTrue(np.allclose(net1_tree_contracted2, net1_tree_contracted))

        # second network
        stn2 = qib.tensor_network.SymbolicTensorNetwork()
        stn2.add_tensor(qib.tensor_network.SymbolicTensor( 2, ( 7, 15),             (11,  4),             "f"))
        stn2.add_tensor(qib.tensor_network.SymbolicTensor( 1, ( 9,  6, 15),         ( 3,  1,  4),         "g007"))
        stn2.add_tensor(qib.tensor_network.SymbolicTensor( 3, ( 6,  4, 17, 15,  9), ( 1, 16,  6,  4,  2), "h"))
        stn2.add_tensor(qib.tensor_network.SymbolicTensor(-1, ( 7,  9,  4, 17,  9), (11,  3, 16,  6,  2), None))  # virtual tensor for open axes
        stn2.add_bond(qib.tensor_network.SymbolicBond( 4, ( 1,  3,  2)))
        stn2.add_bond(qib.tensor_network.SymbolicBond(11, ( 2, -1)))
        stn2.add_bond(qib.tensor_network.SymbolicBond( 3, (-1,  1,)))
        stn2.add_bond(qib.tensor_network.SymbolicBond( 6, ( 3, -1)))
        stn2.add_bond(qib.tensor_network.SymbolicBond(16, (-1,  3)))
        stn2.add_bond(qib.tensor_network.SymbolicBond( 2, ( 3, -1)))
        stn2.add_bond(qib.tensor_network.SymbolicBond( 1, ( 1,  3)))
        self.assertTrue(stn2.is_consistent())
        stn2.rename_bond(1, 0)
        self.assertTrue(stn2.is_consistent())
        ntens2 = stn2.num_tensors
        nbond2 = stn2.num_bonds
        # generate random entries for the tensors in the network
        data_dict2 = { tensor.dataref: rng.normal(size=tensor.shape) for tensor in stn2.tensors.values() if tensor.tid != -1 }
        net2 = qib.tensor_network.TensorNetwork(stn2, data_dict2)
        self.assertTrue(net2.is_consistent())

        # merge networks
        join_axes = [(1, 1), (3, 3), (4, 2), (1, 4)]   # note that axis 1 appears twice
        net1.merge(net2, join_axes)
        self.assertTrue(net1.num_tensors == ntens1 + ntens2)
        self.assertTrue(net1.num_bonds == nbond1 + nbond2 - len(join_axes))
        self.assertTrue(net1.is_consistent())
        bond = net1.net.get_bond(net1.net.get_tensor(8).bids[1])
        self.assertEqual(len(bond.tids), 3)
        self.assertEqual(net1.net.get_tensor(bond.tids[0]).dataref, "c06")
        self.assertEqual(net1.net.get_tensor(bond.tids[1]).dataref, "g007")
        self.assertEqual(net1.net.get_tensor(bond.tids[2]).dataref, "h")
        self.assertEqual(net1.num_open_axes, 4)

        # tensor network representation of a single tensor
        a = rng.normal(size=(5, 1, 2, 3))
        net3 = qib.tensor_network.TensorNetwork.wrap(a, "a")
        self.assertTrue(net3.is_consistent())
        self.assertEqual(net3.num_tensors, 1)
        self.assertEqual(net3.shape, a.shape)
        self.assertTrue(np.array_equal(net3.contract_einsum()[0], a))
        self.assertTrue(np.array_equal(net3.contract_tree(0)[0], a))


if __name__ == "__main__":
    unittest.main()
