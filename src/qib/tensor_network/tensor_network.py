import numpy as np
from typing import Sequence
from qib.tensor_network.symbolic_network import SymbolicTensorNetwork
from qib.tensor_network.contraction_tree import perform_tree_contraction


class TensorNetwork:
    """
    Tensor network, consisting of a symbolic network representation,
    and a dictionary storing the tensor entries addressed by the `dataref`
    member variable of the tensors.
    """
    def __init__(self, net: SymbolicTensorNetwork, data: dict):
        self.net = net
        self.data = data

    @property
    def num_tensors(self) -> int:
        """
        Number of logical tensors.
        """
        return self.net.num_tensors

    @property
    def num_bonds(self) -> int:
        """
        Number of bonds.
        """
        return self.net.num_bonds

    @property
    def num_open_axes(self) -> int:
        """
        Number of open (uncontracted) axes in the network.
        """
        return self.net.num_open_axes

    @property
    def shape(self) -> tuple:
        """
        Logical shape of tensor network (after contracting all bonds).
        """
        return self.net.shape

    def merge(self, other, join_axes: Sequence[tuple]=[]):
        """
        Merge network with another tensor network,
        and join open axes specified as [(openax_self, openax_other), ...].
        """
        self.net.merge(other.net, join_axes)
        # include tensor data from other network
        for k in other.data:
            if k in self.data:
                if not np.array_equal(self.data[k], other.data[k]):
                    raise ValueError(f"tensor data entries for {k} in the to-be joined networks do not match")
        self.data.update(other.data)

    def contract_einsum(self):
        """
        Contract the overall network by a single call of np.einsum.
        """
        tids, tidx, idxout, axes_map = self.net.as_einsum()
        # arguments for call of np.einsum
        args = []
        for i in range(len(tids)):
            tensor = self.net.tensors[tids[i]]
            args.append(self.data[tensor.dataref])
            args.append(tidx[i])
        args.append(idxout)
        return np.einsum(*args), axes_map

    def contract_tree(self, scaffold):
        """
        Contract the overall network based on the
        contraction tree specified by `scaffold`.
        """
        # binary tree contraction of network
        tree = self.net.build_contraction_tree(scaffold)
        # map logical output axes to axes of tree root tensor
        tensor_open_axes = self.net.tensors[-1]
        axes_map = tensor_open_axes.ndim * [-1]
        for i, bid in enumerate(tensor_open_axes.bids):
            bond = self.net.bonds[bid]
            for tid, ax in zip(bond.tids, bond.axes):
                if tid == -1:
                    continue
                if (tid, ax) not in tree.openaxes:
                    raise RuntimeError(f"axis {ax} of tensor {tid} not found among open axes of tree")
                k = tree.trackaxes[tree.openaxes.index((tid, ax))]
                if axes_map[i] == -1:
                    axes_map[i] = k
                else:
                    # consistency check
                    if axes_map[i] != k:
                        raise RuntimeError(f"inconsistency when tracking open axis {i} of network to tree root tensor")
            if axes_map[i] == -1:
                raise RuntimeError(f"cannot track open axis {i} of network to tree root tensor")
        # permute tree root axes to match logical output axes as far as possible
        sort_indices = tree.ndim * [-1]
        c = 0
        for i in range(len(axes_map)):
            if sort_indices[axes_map[i]] == -1:
                # use next available index
                sort_indices[axes_map[i]] = c
                c += 1
        assert c == tree.ndim
        tree.permute_axes(np.argsort(sort_indices))
        axes_map = [sort_indices[k] for k in axes_map]
        # perform contraction
        tensor_dict = { tensor.tid: self.data[tensor.dataref] for tensor in self.net.tensors.values() if tensor.tid != -1 }
        cnt = perform_tree_contraction(tree, tensor_dict)
        # return contracted tensor, axes map and tree
        return cnt, axes_map, tree

    def is_consistent(self, verbose=False) -> bool:
        """
        Perform an internal consistency check,
        e.g., whether the bond ID specified by any tensor actually exist.
        """
        if not self.net.is_consistent(verbose):
            return False
        for tensor in self.net.tensors.values():
            if tensor.tid == -1:
                continue
            if tensor.dataref not in self.data:
                if verbose: print(f"Consistency check failed: dataref {tensor.dataref} not in data dictionary.")
                return False
            if np.shape(self.data[tensor.dataref]) != tensor.shape:
                if verbose: print(f"Consistency check failed: shape of dataref {tensor.dataref} does not match shape of symbolic tensor {tensor.tid}.")
                return False
        return True
