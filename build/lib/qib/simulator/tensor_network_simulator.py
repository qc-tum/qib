import numpy as np
from typing import Sequence
from qib.simulator import Simulator
from qib.tensor_network import SymbolicTensor, SymbolicBond, SymbolicTensorNetwork, TensorNetwork
from qib.tensor_network.tensor_network import to_full_tensor
from qib.circuit import Circuit
from qib.field import Field


class TensorNetworkSimulator(Simulator):
    """
    Tensor network simulator, contracting a circuit interpreted as tensor network.
    """

    def run(self, circ: Circuit, fields: Sequence[Field], description):
        """
        Run a quantum circuit simulation.
        """
        # use |0> states as input
        init_stn = SymbolicTensorNetwork()
        init_data = {}
        local_dims = []
        for field in fields:
            for j in range(field.lattice.nsites):
                dataref = "|0>_" + str(field.local_dim)
                i = len(local_dims)
                init_stn.add_tensor(SymbolicTensor(i, (field.local_dim,), (i,), dataref))
                if dataref not in init_data:
                    ket0 = np.zeros(field.local_dim)
                    ket0[0] = 1
                    init_data[dataref] = ket0
                local_dims.append(field.local_dim)
        # virtual tensor for open axes
        init_stn.add_tensor(SymbolicTensor(-1, local_dims, list(range(len(local_dims))), None))
        # add bonds to specify open axes
        for i in range(len(local_dims)):
            init_stn.add_bond(SymbolicBond(i, (-1, i)))
        init_net = TensorNetwork(init_stn, init_data)
        assert init_net.is_consistent()
        assert init_net.num_open_axes == len(local_dims)

        net = circ.as_tensornet(fields)
        net.num_open_axes == 2*len(local_dims)
        # merge with init_net for initial |0> states
        net.merge(init_net, [(len(local_dims) + i, i) for i in range(len(local_dims))])

        # for simplicity, output is full statevector
        # TODO: tensor network simulation for computing expectation values of observables
        tensor, axes_map = net.contract_einsum()

        # in most use-cases, output axes are not duplicate,
        # so returning a tensor here for conceptual simplicity
        return to_full_tensor(tensor, axes_map)
