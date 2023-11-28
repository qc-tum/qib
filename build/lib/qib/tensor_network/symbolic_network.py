import copy
from typing import Sequence
from qib.tensor_network.contraction_tree import ContractionTreeNode


class SymbolicTensor:
    """
    Symbolic tensor, storing a unique ID, its shape, connected bond IDs
    (representing contractions with other tensors)
    and a user-provided reference to the data of the tensor.
    """
    def __init__(self, tid: int, shape: Sequence[int], bids: Sequence[int], dataref):
        if len(shape) != len(bids):
            raise ValueError("number of dimensions must be equal to number of connected bond IDs")
        self.tid     = tid
        self.shape   = tuple(shape)
        self.bids    = list(bids)
        self.dataref = dataref

    @property
    def ndim(self) -> int:
        """
        Number of dimensions (degree) of the tensor.
        """
        return len(self.shape)

    def transpose(self, axes=None):
        """
        Transpose the axes of the tensor, analogous to numpy.transpose.
        Tensor is updated in-place.
        """
        if axes is None:
            axes = reversed(range(self.ndim))
        if len(set(axes)) != len(axes):
            raise ValueError(f"axes = {axes} is not a valid permutation")
        self.shape = tuple(self.shape[ax] for ax in axes)
        self.bids  =       [self.bids[ax] for ax in axes]
        # enable chaining
        return self


class SymbolicBond:
    """
    Symbolic bond in a tensor network, e.g., an edge or multi-edge between tensors.
    The corresponding axes of the respective tensor can be inferred from the
    bond ID index stored in the tensor.
    A bond can be connected to several axes of the same tensor;
    in this case, the same tensor ID is referenced multiple times by the bond.

    Member variables:
        bid:  bond ID
        tids: tensor IDs
    """
    def __init__(self, bid: int, tids: Sequence[int]):
        if len(tids) < 2:
            raise ValueError("bond must reference at least two tensors")
        self.bid = bid
        self.tids = sorted(list(tids))


class SymbolicTensorNetwork:
    """
    Symbolic tensor network, storing a list of tensors and bonds.
    Open (uncontracted) axes in the network are represented by the legs of a virtual tensor with ID -1.
    """
    def __init__(self, tensors: Sequence[SymbolicTensor]=None, bonds: Sequence[SymbolicBond]=None):
        if tensors is None:
            tensors = []
        if bonds is None:
            bonds = []
        # dictionary of tensors
        self.tensors = {}
        for tensor in tensors:
            self.add_tensor(tensor)
        # dictionary of bonds
        self.bonds = {}
        for bond in bonds:
            self.add_bond(bond)

    @property
    def num_tensors(self) -> int:
        """
        Number of logical tensors.
        """
        if -1 not in self.tensors:
            raise RuntimeError("network requires a virtual tensor with ID -1 for the open axes")
        # virtual tensor for open axes does not count towards logical tensors
        return len(self.tensors) - 1

    @property
    def num_open_axes(self) -> int:
        """
        Number of open (uncontracted) axes in the network.
        """
        if -1 not in self.tensors:
            raise RuntimeError("network requires a virtual tensor with ID -1 for the open axes")
        return self.tensors[-1].ndim

    @property
    def shape(self) -> tuple:
        """
        Logical shape of tensor network (after contracting all bonds).
        """
        if -1 not in self.tensors:
            raise RuntimeError("network requires a virtual tensor with ID -1 for the open axes")
        return self.tensors[-1].shape

    def transpose(self, axes=None):
        """
        Logically transpose the network, i.e., the virtual output tensor of the network.
        """
        if -1 not in self.tensors:
            raise RuntimeError("network requires a virtual tensor with ID -1 for the open axes")
        self.tensors[-1].transpose(axes)
        # enable chaining
        return self

    def has_tensor(self, tid: int) -> bool:
        """
        Whether the tensor with ID `tid` exists.
        """
        return tid in self.tensors

    def get_tensor(self, tid: int) -> SymbolicTensor:
        """
        Get the tensor with ID `tid`.
        """
        return self.tensors[tid]

    def add_tensor(self, tensor: SymbolicTensor):
        """
        Add a tensor to the network.
        """
        if tensor.tid in self.tensors:
            raise ValueError(f"tensor with ID {tensor.tid} already exists")
        self.tensors[tensor.tid] = tensor

    def rename_tensor(self, tid_cur: int, tid_new: int):
        """
        Rename tensor ID `tid_cur` -> `tid_new`.
        """
        if tid_cur not in self.tensors:
            raise ValueError(f"tensor with ID {tid_cur} does not exist")
        if tid_new in self.tensors:
            raise ValueError(f"tensor with ID {tid_new} already exists")
        tensor = self.tensors.pop(tid_cur)
        assert tensor.tid == tid_cur
        for bid in tensor.bids:
            bond = self.bonds[bid]
            for i in range(len(bond.tids)):
                if bond.tids[i] == tid_cur:
                    bond.tids[i] = tid_new
            bond.tids.sort()
        tensor.tid = tid_new
        self.tensors[tid_new] = tensor

    def merge_tensors(self, tid1: int, tid2: int):
        """
        Merge tensors with IDs `tid1` and `tid2`.
        The resulting tensor inherits ID `tid1`.
        """
        if tid1 == tid2:
            return
        tensor1 = self.tensors[tid1]
        tensor2 = self.tensors.pop(tid2)
        for bid in tensor2.bids:
            bond = self.bonds[bid]
            for i in range(len(bond.tids)):
                if bond.tids[i] == tid2:
                    bond.tids[i] = tid1
            bond.tids.sort()
        tensor1.shape += tensor2.shape
        tensor1.bids  += tensor2.bids

    def tensor_ids(self) -> list:
        """
        Return the list of tensor IDs in the network (without virtual tensor ID -1).
        """
        if -1 not in self.tensors:
            raise RuntimeError("network requires a virtual tensor with ID -1 for the open axes")
        tids = sorted(list(self.tensors.keys()))
        tids.remove(-1)
        return tids

    @property
    def num_bonds(self) -> int:
        """
        Number of bonds.
        """
        return len(self.bonds)

    def has_bond(self, bid: int) -> bool:
        """
        Whether the bond with ID `bid` exists.
        """
        return bid in self.bonds

    def get_bond(self, bid: int) -> SymbolicBond:
        """
        Get the bond with ID `bid`.
        """
        return self.bonds[bid]

    def add_bond(self, bond: SymbolicBond):
        """
        Add a bond to the network, and create references to this bond in tensors.
        """
        if bond.bid in self.bonds:
            raise ValueError(f"bond with ID {bond.bid} already exists")
        self.bonds[bond.bid] = bond

    def generate_bonds(self):
        """
        Generate bonds based on the information stored in the tensors.
        """
        if self.bonds:
            raise RuntimeError("expecting empty bond collection before generating bonds")
        bidlist = []
        for tensor in self.tensors.values():
            bidlist += tensor.bids
        # unique bond IDs
        bidlist = sorted(list(set(bidlist)))
        for bid in bidlist:
            tids = []
            for tensor in self.tensors.values():
                # note: same tensor ID can appear multiple times
                for b in tensor.bids:
                    if b == bid:
                        tids.append(tensor.tid)
            self.add_bond(SymbolicBond(bid, tids))
        # enable chaining
        return self

    def rename_bond(self, bid_cur: int, bid_new: int):
        """
        Rename bond ID `bid_cur` -> `bid_new`.
        """
        if bid_cur not in self.bonds:
            raise ValueError(f"bond with ID {bid_cur} does not exist")
        if bid_new in self.bonds:
            raise ValueError(f"bond with ID {bid_new} already exists")
        bond = self.bonds.pop(bid_cur)
        assert bond.bid == bid_cur
        bond.bid = bid_new
        self.bonds[bid_new] = bond
        # update bond references in tensors
        for tid in bond.tids:
            tensor = self.tensors[tid]
            for ax in range(len(tensor.bids)):
                if tensor.bids[ax] == bid_cur:
                    tensor.bids[ax] = bid_new

    def merge_bonds(self, bid1: int, bid2: int):
        """
        Merge bonds with IDs `bid1` and `bid2`.
        The resulting bond inherits ID `bid1`.
        """
        if bid1 == bid2:
            return
        bond1 = self.bonds[bid1]
        bond2 = self.bonds.pop(bid2)
        for tid in bond2.tids:
            bond1.tids.append(tid)
            # update bond reference in tensor
            tensor = self.tensors[tid]
            for ax in range(len(tensor.bids)):
                if tensor.bids[ax] == bid2:
                    tensor.bids[ax] = bid1
        bond1.tids.sort()

    def get_bond_axes(self, bid: int):
        """
        Get the axes corresponding to the ordered tensor IDs referenced by the bond.
        """
        bond = self.bonds[bid]
        assert bond.bid == bid
        axes = len(bond.tids) * [-1]
        for i in range(len(bond.tids)):
            # j-th occurrence of current tensor ID
            j = bond.tids[:i].count(bond.tids[i])
            tensor = self.tensors[bond.tids[i]]
            for ax in range(len(tensor.bids)):
                if tensor.bids[ax] == bid:
                    if j == 0:
                        axes[i] = ax
                        break
                    j -= 1
        assert all(ax >= 0 for ax in axes)
        return axes

    def merge(self, other, join_axes: Sequence[tuple]=None):
        """
        Merge network with another symbolic tensor network,
        and join open axes specified as [(openax_self, openax_other), ...].
        """
        if join_axes is None:
            join_axes = []
        for joinax in join_axes:
            if joinax[0] < 0 or joinax[0] >= self.num_open_axes:
                raise ValueError(f"to-be joined open axis index {joinax[0]} of first network out of range")
            if joinax[1] < 0 or joinax[1] >= other.num_open_axes:
                raise ValueError(f"to-be joined open axis index {joinax[1]} of second network out of range")
        num_open_axes_orig = self.num_open_axes
        # require a deep copy since the IDs in the 'other' network might change
        other = copy.deepcopy(other)
        # ensure that tensor IDs in the two networks are disjoint
        shared_tids = self.tensors.keys() & other.tensors.keys()
        tmp_open_tid = -1
        next_tid = max(self.tensors.keys() | other.tensors.keys(), default=0) + 1
        for tid in shared_tids:
            other.rename_tensor(tid, next_tid)
            if tid == -1:
                tmp_open_tid = next_tid
            next_tid += 1
        # ensure that bond IDs in the two networks are disjoint
        shared_bids = self.bonds.keys() & other.bonds.keys()
        next_bid = max(self.bonds.keys() | other.bonds.keys(), default=0) + 1
        for bid in shared_bids:
            other.rename_bond(bid, next_bid)
            next_bid += 1
        # include tensors from other network
        self.tensors.update(other.tensors)
        # include bonds from other network
        self.bonds.update(other.bonds)
        # merge virtual tensors for open axes
        self.merge_tensors(-1, tmp_open_tid)
        # join specified open axes
        tensor_open_axes = self.tensors[-1]
        axes_map = list(range(tensor_open_axes.ndim))
        for joinax in join_axes:
            self.merge_bonds(tensor_open_axes.bids[joinax[0]],
                             tensor_open_axes.bids[num_open_axes_orig + joinax[1]])
            # record remaining axes;
            # same open axis could be joined with several open axes in other network
            if joinax[0] in axes_map:
                axes_map.remove(joinax[0])
            if num_open_axes_orig + joinax[1] in axes_map:
                axes_map.remove(num_open_axes_orig + joinax[1])
        # to-be deleted open axes
        del_axes = [i*num_open_axes_orig + joinax[i] for joinax in join_axes for i in range(2)]
        # remove to-be deleted open axes references from bonds
        for delax in del_axes:
            bid = tensor_open_axes.bids[delax]
            bond = self.bonds[bid]
            # remove reference to axis from bond
            if -1 in bond.tids:
                # note: removing only one occurrence of -1 from bonds.tids
                bond.tids.remove(-1)
            assert len(bond.tids) >= 2
        # remove to-be joined axes from tensor
        tensor_open_axes.shape = tuple(tensor_open_axes.shape[i] for i in axes_map)
        tensor_open_axes.bids  = [tensor_open_axes.bids[i] for i in axes_map]
        # enable chaining
        return self

    def build_contraction_tree(self, scaffold) -> ContractionTreeNode:
        """
        Build the contraction tree based on the contraction ordering in `scaffold`,
        which is a recursively nested list of IDs to specify the tree, e.g.,
        scaffold = [[ta, tb], [[tc, td], te]].
        """
        # search for next available tensor ID
        max_tid = max(self.tensors.keys(), default=0)
        return self._build_contraction_tree(scaffold, max_tid + 1)

    def _build_contraction_tree(self, scaffold, next_tid) -> ContractionTreeNode:
        """
        Recursively build the contraction tree,
        starting from `next_tid` for generating intermediate tensor IDs.
        Returns a ContractionTreeNode as root of the tree.
        """
        if isinstance(scaffold, int):   # leaf node
            if scaffold == -1:
                raise ValueError("cannot use virtual tensor for open axes in contraction tree")
            tensor = self.tensors[scaffold]
            openaxes = [(scaffold, i) for i in range(tensor.ndim)]
            return ContractionTreeNode(scaffold, None, [], None, [], list(range(tensor.ndim)), openaxes, list(range(tensor.ndim)))
        assert isinstance(scaffold, Sequence), "invalid `scaffold` argument"
        assert len(scaffold) == 2, "`scaffold` must specify pairwise contractions"
        # generate child nodes
        nL = self._build_contraction_tree(scaffold[0], next_tid)
        if nL.tid >= next_tid: next_tid = nL.tid + 1
        nR = self._build_contraction_tree(scaffold[1], next_tid)
        if nR.tid >= next_tid: next_tid = nR.tid + 1
        assert not set(nL.openaxes).intersection(set(nR.openaxes)), "open axes of left and right subtrees must be disjoint"
        # find bonds attached to the left or right subtrees
        bidlist = []
        bmaplist = []
        openaxes = nL.openaxes + nR.openaxes
        for od in nL.openaxes + nR.openaxes:
            bid = self.tensors[od[0]].bids[od[1]]
            # reference to bond can appear multiple times due to multi-edge bonds
            if bid in bidlist:
                continue
            bidlist.append(bid)
            bond = self.bonds[bid]
            bond_axes = self.get_bond_axes(bid)
            bmap = len(bond.tids) * [None]
            for i in range(len(bond.tids)):
                ta = (bond.tids[i], bond_axes[i])
                if ta in nL.openaxes:
                    bmap[i] = ("L", nL.trackaxes[nL.openaxes.index(ta)])
                elif ta in nR.openaxes:
                    bmap[i] = ("R", nR.trackaxes[nR.openaxes.index(ta)])
            bmaplist.append(bmap)
            # if bond is fully contracted (no upstream connections)
            if all(bmap):
                for ta in zip(bond.tids, bond_axes):
                    openaxes.remove(ta)
        # tensor degrees
        deg = [len(nL.idxout), len(nR.idxout)]
        # contraction indices
        idxL = list(range(deg[0]))
        idxR = list(range(deg[0], deg[0] + deg[1]))
        idxout = idxL + idxR
        for bmap in bmaplist:
            # whether corresponding bond is fully contracted (no upstream connections)
            fully_contracted = all(bmap)
            # shared np.einsum index
            j = -1
            for bm in bmap:
                if bm is None:
                    continue
                if bm[0] == "L":
                    k = bm[1]
                    if j != -1:
                        if idxL[k] in idxout and idxL[k] != j:
                            idxout.remove(idxL[k])
                        idxL[k] = j
                    else:
                        if fully_contracted:
                            idxout.remove(idxL[k])
                        j = idxL[k]
                elif bm[0] == "R":
                    k = bm[1]
                    if j != -1:
                        if idxR[k] in idxout and idxR[k] != j:
                            idxout.remove(idxR[k])
                        idxR[k] = j
                    else:
                        if fully_contracted:
                            idxout.remove(idxR[k])
                        j = idxR[k]
        # construct `trackaxes`
        trackaxes = len(openaxes) * [None]
        for i, ta in enumerate(openaxes):
            if ta in nL.openaxes:
                k = nL.trackaxes[nL.openaxes.index(ta)]
                trackaxes[i] = idxout.index(idxL[k])
            elif ta in nR.openaxes:
                k = nR.trackaxes[nR.openaxes.index(ta)]
                trackaxes[i] = idxout.index(idxR[k])
            else:
                assert False
        return ContractionTreeNode(next_tid, nL, idxL, nR, idxR, idxout, openaxes, trackaxes)

    def as_einsum(self) -> tuple:
        """
        Convert the contractions in the network to an `numpy.einsum` argument list,
        for a single call of `numpy.einsum`. The ordering of the output axes
        follows the virtual tensor for the open axes.

        Returns:
            tids, tidx, idxout, axes_map: tensor IDs and index argument list for `numpy.einsum`,
                and map from logical open axes to axes of output tensor of `numpy.einsum`
                (einsum does not support an output index appearing multiple times)
        """
        # tensor IDs; ensuring that ID -1 (for virtual open axes tensor) is the last entry
        max_tid = max(self.tensors.keys(), default=0)
        tids = sorted(list(self.tensors.keys()), key=lambda tid: max_tid+1 if tid==-1 else tid)
        assert tids[-1] == -1
        # generate continuous indices for all tensors
        tidx = []
        maxidx = 0
        for tid in tids:
            ndim = self.tensors[tid].ndim
            tidx.append(list(range(maxidx, maxidx + ndim)))
            maxidx += ndim
        # identify to-be contracted indices (or shared indices for output)
        for bond in self.bonds.values():
            it = [tids.index(tid) for tid in bond.tids]
            bond_axes = self.get_bond_axes(bond.bid)
            imin = min((tidx[i][ax] for i, ax in zip(it, bond_axes)), default=0)
            for i, ax in zip(it, bond_axes):
                tidx[i][ax] = imin
        # condense indices
        idxmap = maxidx * [-1]
        c = 0
        for i in range(len(tidx)):
            for j in range(len(tidx[i])):
                if idxmap[tidx[i][j]] == -1:
                    # use next available index
                    idxmap[tidx[i][j]] = c
                    c += 1
                tidx[i][j] = idxmap[tidx[i][j]]
        # indices for tensor -1 are the output indices
        idxout = tidx[-1]
        tids = tids[:-1]
        tidx = tidx[:-1]
        # einsum does not support an output index appearing multiple times
        idxout_logical = idxout.copy()
        # keep only first occurrence
        idxout = [i for k, i in enumerate(idxout) if i not in idxout[:k]]
        # record first occurrence of each index
        axes_map = [idxout.index(i) for i in idxout_logical]
        return tids, tidx, idxout, axes_map

    def is_consistent(self, verbose=False) -> bool:
        """
        Perform an internal consistency check,
        e.g., whether the bond ID specified by any tensor actually exist.
        """
        if -1 not in self.tensors:
            if verbose: print("Consistency check failed: network requires a virtual tensor with ID -1 for the open bonds.")
            return False
        for k, tensor in self.tensors.items():
            if k != tensor.tid:
                if verbose: print(f"Consistency check failed: dictionary key {k} does not match tensor ID {tensor.tid}.")
                return False
            for bid in tensor.bids:
                # bond with ID 'bid' must exist
                if bid not in self.bonds:
                    if verbose: print(f"Consistency check failed: bond with ID {bid} referenced by tensor {k} does not exist.")
                    return False
                # bond must refer back to tensor and corresponding axis
                bond = self.bonds[bid]
                if tensor.tid not in bond.tids:
                    if verbose: print(f"Consistency check failed: bond with ID {bid} does not refer to tensor {tensor.tid}.")
                    return False
        for k, bond in self.bonds.items():
            if k != bond.bid:
                if verbose: print(f"Consistency check failed: dictionary key {k} does not match bond ID {bond.bid}.")
                return False
            if len(bond.tids) < 2:
                if verbose: print(f"Consistency check failed: bond {bond.bid} references {len(bond.tids)} tensor(s), should reference at least two.")
                return False
            dims = []
            bond_axes = self.get_bond_axes(bond.bid)
            for i in range(len(bond.tids)):
                if (bond.tids[i], bond_axes[i]) in zip(bond.tids[:i], bond_axes[:i]):
                    if verbose: print(f"Consistency check failed: axis {bond_axes[i]} of tensor {bond.tids[i]} referenced twice by bond {k}.")
                    return False
            for tid, ax in zip(bond.tids, bond_axes):
                # tensor with ID 'tid' must exist
                if tid not in self.tensors:
                    if verbose: print(f"Consistency check failed: tensor with ID {tid} referenced by bond {k} does not exist.")
                    return False
                # axis 'ax' of tensor must refer back to bond
                tensor = self.tensors[tid]
                if len(tensor.bids) <= ax:
                    if verbose: print(f"Consistency check failed: tensor {tensor.tid} does not have a {ax}-th axis.")
                    return False
                if tensor.bids[ax] != bond.bid:
                    if verbose: print(f"Consistency check failed: axis {ax} of tensor {tensor.tid} does not refer to bond {bond.bid}.")
                    return False
                dims.append(tensor.shape[ax])
            if dims:
                if not all(d == dims[0] for d in dims):
                    if verbose: print(f"Consistency check failed: axes dimensions of bond {bond.bid} are not all the same.")
                    return False
        return True
