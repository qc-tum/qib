import unittest
import qib


class TestTensorNetwork(unittest.TestCase):

    def test_symbolic_network(self):
        """
        Test symbolic network functionality.
        """
        # first network
        net1 = qib.tensor_network.SymbolicTensorNetwork()
        net1.add_tensor(qib.tensor_network.SymbolicTensor( 3, ( 3,  5,  6,  2), 0))
        net1.add_tensor(qib.tensor_network.SymbolicTensor( 1, ( 7, 19,  2,  5), 0))
        net1.add_tensor(qib.tensor_network.SymbolicTensor( 7, ( 7,  9,  3, 17, 17), 0.06))  # special dataref, used as marker
        net1.add_tensor(qib.tensor_network.SymbolicTensor( 0, (13,  7,  3), 0))
        net1.add_tensor(qib.tensor_network.SymbolicTensor( 5, (13, 11,  6), 0))
        net1.add_tensor(qib.tensor_network.SymbolicTensor(-1, (11,  9,  7, 17, 19), 0))     # dummy tensor for open axes
        net1.add_bond(qib.tensor_network.SymbolicBond(-7, ( 3,  1),     ( 1,  3)))
        net1.add_bond(qib.tensor_network.SymbolicBond( 3, ( 0,  7,  3), ( 2,  2,  0)))
        net1.add_bond(qib.tensor_network.SymbolicBond(14, ( 1,  3),     ( 2,  3)))
        net1.add_bond(qib.tensor_network.SymbolicBond(-3, ( 5,  3),     ( 2,  2)))
        net1.add_bond(qib.tensor_network.SymbolicBond( 0, (-1,  0,),    ( 2,  1,)))
        net1.add_bond(qib.tensor_network.SymbolicBond( 2, ( 0,  5),     ( 0,  0)))
        net1.add_bond(qib.tensor_network.SymbolicBond(11, ( 5, -1),     ( 1,  0)))
        net1.add_bond(qib.tensor_network.SymbolicBond(13, ( 7, -1),     ( 1,  1)))
        net1.add_bond(qib.tensor_network.SymbolicBond( 5, ( 7, -1,  7), ( 3,  3,  4)))
        net1.add_bond(qib.tensor_network.SymbolicBond( 6, ( 1,  7),     ( 0,  0)))
        net1.add_bond(qib.tensor_network.SymbolicBond(17, ( 1, -1),     ( 1,  4)))
        self.assertTrue(net1.is_consistent())
        ntens1 = net1.num_tensors
        nbond1 = net1.num_bonds
        net1.rename_tensor( 7, 8)
        self.assertTrue(net1.is_consistent())
        net1.rename_bond(14, 10)
        self.assertTrue(net1.is_consistent())
        # second network
        net2 = qib.tensor_network.SymbolicTensorNetwork()
        net2.add_tensor(qib.tensor_network.SymbolicTensor( 2, ( 7, 15), 0))
        net2.add_tensor(qib.tensor_network.SymbolicTensor( 1, ( 9,  6, 15), 0.07))  # special dataref, used as marker
        net2.add_tensor(qib.tensor_network.SymbolicTensor( 3, ( 6, 19, 17, 15,  9), 0.08))
        net2.add_tensor(qib.tensor_network.SymbolicTensor(-1, ( 7,  9, 19, 17,  9), 0))     # dummy tensor for open axes
        net2.add_bond(qib.tensor_network.SymbolicBond( 4, ( 1,  3,  2), ( 2,  3,  1)))
        net2.add_bond(qib.tensor_network.SymbolicBond(11, ( 2, -1),     ( 0,  0)))
        net2.add_bond(qib.tensor_network.SymbolicBond( 3, (-1,  1,),    ( 1,  0,)))
        net2.add_bond(qib.tensor_network.SymbolicBond( 6, ( 3, -1),     ( 2,  3)))
        net2.add_bond(qib.tensor_network.SymbolicBond(16, (-1,  3),     ( 2,  1,)))
        net2.add_bond(qib.tensor_network.SymbolicBond( 2, ( 3, -1),     ( 4,  4)))
        net2.add_bond(qib.tensor_network.SymbolicBond( 1, ( 1,  3),     ( 1,  0)))
        self.assertTrue(net2.is_consistent())
        net2.rename_bond(1, 0)
        self.assertTrue(net2.is_consistent())
        ntens2 = net2.num_tensors
        nbond2 = net2.num_bonds
        # merge networks
        join_axes = [(1, 1), (3, 3), (4, 2), (1, 4)]   # note that axis 1 appears twice
        net1.merge(net2, join_axes)
        self.assertTrue(net1.num_tensors == ntens1 + ntens2)
        self.assertTrue(net1.num_bonds == nbond1 + nbond2 - len(join_axes))
        self.assertTrue(net1.is_consistent())
        bond = net1.get_bond(net1.get_tensor(8).bids[1])
        self.assertEqual(len(bond.tids), 3)
        self.assertEqual(net1.get_tensor(bond.tids[0]).dataref, 0.06)
        self.assertEqual(net1.get_tensor(bond.tids[1]).dataref, 0.07)
        self.assertEqual(net1.get_tensor(bond.tids[2]).dataref, 0.08)
        self.assertEqual(net1.num_open_axes, 3)


if __name__ == "__main__":
    unittest.main()
