import numpy as np


class ContractionTreeNode:
    """
    Node in a binary contraction tree,
    following numpy.einsum's index specification convention
    for describing the contraction.

    Member variables:
      * tid:            index of new tensor after contraction
      * idxL:           contraction indices for left child node tensor
      * idxR:           contraction indices for right child node tensor
      * idxout:         contraction indices for new (output) tensor
      * openaxes:       list of open axes of the leaf tensors, of the form [(tid, ax), ...]
      * trackaxes:      i-th open axis corresponds to trackaxes[i]-th leg of node tensor;
                        in general not bijective due to partial contractions from multi-edges
      * parent:         parent node
      * children:       [nL, nR], left and right child nodes
    """
    def __init__(self, tid: int, nL, idxL, nR, idxR, idxout, openaxes, trackaxes):
        self.tid = tid
        # indices as used by numpy.einsum; `idxL` and `idxR` can be empty in case this is a leaf node
        self.idxL = list(idxL)
        self.idxR = list(idxR)
        self.idxout = list(idxout)
        self.openaxes = list(openaxes)
        self.trackaxes = list(trackaxes)
        self.parent = None
        self.children = [nL, nR]
        if nL: nL.parent = self
        if nR: nR.parent = self

    @property
    def is_leaf(self):
        """
        Whether this node is a leaf.
        """
        return not any(self.children)

    @property
    def ndim(self):
        """
        Logical number of dimensions (degree) of the tensor represented by the node.
        """
        return len(self.idxout)

    def permute_axes(self, sort_indices):
        """
        Permute the axes of the tensor represented by the node.
        """
        if len(sort_indices) != len(self.idxout):
            raise ValueError(f"`sort_indices` must be an index sequence of length {len(self.idxout)}")
        self.idxout = [self.idxout[i] for i in sort_indices]
        invsort = np.argsort(sort_indices)
        self.trackaxes = [invsort[k] for k in self.trackaxes]
        if self.parent:
            parent = self.parent
            if self == parent.children[0]:      # whether left child
                parent.idxL = [parent.idxL[i] for i in sort_indices]
            elif self == parent.children[1]:    # whether right child
                parent.idxR = [parent.idxR[i] for i in sort_indices]
            else:
                assert False, "node not found among children of its parent"


def perform_tree_contraction(node: ContractionTreeNode, tensor_dict) -> np.array:
    """
    Perform contraction as specified by contraction tree with root `node`.
    """
    if node.is_leaf:
        return tensor_dict[node.tid]
    assert node.children[0] and node.children[1], "both child nodes must be set"
    tL = perform_tree_contraction(node.children[0], tensor_dict)
    tR = perform_tree_contraction(node.children[1], tensor_dict)
    return np.einsum(tL, node.idxL, tR, node.idxR, node.idxout)
