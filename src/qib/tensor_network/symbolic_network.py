import copy
from typing import Sequence


class SymbolicTensor:
    """
    Symbolic tensor, storing a unique ID, its shape, connected bond IDs
    (representing open tensor legs or contractions with other tensors)
    and a user-provided reference to the data of the tensor.
    """
    def __init__(self, tid: int, shape: Sequence[int], dataref, bids: Sequence[int]=None):
        if bids is None:
            # dummy bond IDs
            bids = len(shape) * [-1]
        if len(shape) != len(bids):
            raise ValueError("number of dimensions must be equal to number of connected bond IDs")
        self.tid     = tid
        self.shape   = tuple(shape)
        self.bids    = list(bids)
        self.dataref = dataref

    @property
    def ndim(self):
        """
        Number of dimensions (degree) of the tensor.
        """
        return len(self.shape)


class SymbolicBond:
    """
    Symbolic bond in a tensor network, e.g., an edge between two tensors,
    an open (uncontracted) tensor leg, or a multi-edge.

    Member variables:
        bid:    bond ID
        tids:   tensor IDs
        axes:   corresponding axes of the respective tensors
    """
    def __init__(self, bid: int, tids: Sequence[int], axes: Sequence[int]):
        if len(tids) != len(axes):
            raise ValueError("number of tensor IDs must match number of axes")
        self.bid = bid
        self.tids = list(tids)
        self.axes = list(axes)
        self.sort()

    def sort(self):
        """
        Sort tensor and axes indices. (Logically the same bond.)
        """
        s = sorted(zip(self.tids, self.axes))
        self.tids = [x[0] for x in s]
        self.axes = [x[1] for x in s]
        # enable chaining
        return self


class SymbolicTensorNetwork:
    """
    Symbolic tensor network, storing a list of tensors and bonds.
    """
    def __init__(self, tensors: Sequence[SymbolicTensor]=[], bonds: Sequence[SymbolicBond]=[]):
        # dictionary of tensors
        self.tensors = {}
        for tensor in tensors:
            self.add_tensor(tensor)
        # dictionary of bonds
        self.bonds = {}
        for bond in bonds:
            self.add_bond(bond)
        # dictionary of tagged bonds
        self.btags = {}

    @property
    def num_tensors(self):
        """
        Number of tensors.
        """
        return len(self.tensors)

    def has_tensor(self, tid: int):
        """
        Whether the tensor with ID `tid` exists.
        """
        return tid in self.tensors

    def get_tensor(self, tid: int):
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
            bond.sort()
        tensor.tid = tid_new
        self.tensors[tid_new] = tensor

    @property
    def num_bonds(self):
        """
        Number of bonds.
        """
        return len(self.bonds)

    def has_bond(self, bid: int):
        """
        Whether the bond with ID `bid` exists.
        """
        return bid in self.bonds

    def get_bond(self, bid: int):
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
        # create bond references in tensors
        for tid, ax in zip(bond.tids, bond.axes):
            if tid not in self.tensors:
                raise ValueError(f"tensor {tid} referenced by bond {bond.bid} does not exist")
            tensor = self.tensors[tid]
            if len(tensor.bids) <= ax:
                raise ValueError(f"tensor {tensor.tid} does not have a {ax}-th axis.")
            tensor.bids[ax] = bond.bid

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
        for tid, ax in zip(bond.tids, bond.axes):
            assert self.tensors[tid].bids[ax] == bid_cur
            self.tensors[tid].bids[ax] = bid_new
        # update bond references in tags
        for tbids in self.btags.values():
            for i in range(len(tbids)):
                if tbids[i] == bid_cur:
                    tbids[i] = bid_new

    def get_tag(self, tkey):
        """
        Get the tagged bonds under key `tkey`.
        """
        return self.btags[tkey]

    def add_tag(self, tkey, tbids: Sequence[int]):
        """
        Add a tag for a list of existing bond IDs.
        """
        if tkey in self.btags:
            raise ValueError(f"tag {tkey} already exists")
        # bond IDs must already exist in the network
        for bid in tbids:
            if bid not in self.bonds:
                raise ValueError(f"bond with ID {bid} does not exist")
        self.btags[tkey] = list(tbids)

    def merge(self, other, join_bonds: Sequence[tuple]=[]):
        """
        Merge network with another symbolic tensor network,
        and join bonds specified as [(bid_self, bid_other), ...].
        The keys for the bond tags in the two networks must be unique.
        """
        # require a deep copy since the IDs in the 'other' network might change
        other = copy.deepcopy(other)
        # ensure that tensor IDs in the two networks are disjoint
        shared_tids = self.tensors.keys() & other.tensors.keys()
        if shared_tids:
            next_tid = max(self.tensors.keys() | other.tensors.keys()) + 1
            for tid in shared_tids:
                other.rename_tensor(tid, next_tid)
                next_tid += 1
        # include tensors from other network
        self.tensors.update(other.tensors)
        # to-be joined bond IDs in other network
        join_bids_other = [jb[1] for jb in join_bonds]
        # ensure that non-joined bond IDs in the two networks are disjoint
        shared_bids = self.bonds.keys() & other.bonds.keys()
        if shared_bids:
            next_bid = max(self.bonds.keys() | other.bonds.keys()) + 1
            for bid in shared_bids:
                if bid not in join_bids_other:
                    other.rename_bond(bid, next_bid)
                    next_bid += 1
        # join bonds
        for jb in join_bonds:
            bond_dst = self.bonds[jb[0]]
            bond_src = other.bonds.pop(jb[1])
            for tid, ax in zip(bond_src.tids, bond_src.axes):
                bond_dst.tids.append(tid)
                bond_dst.axes.append(ax)
                # update bond reference in tensor
                assert self.tensors[tid].bids[ax] == jb[1]
                self.tensors[tid].bids[ax] = jb[0]
            bond_dst.sort()
            # update bond reference in tags
            for tbids in other.btags.values():
                for i in range(len(tbids)):
                    if tbids[i] == jb[1]:
                        tbids[i] = jb[0]
        # include remaining bonds from other network
        self.bonds.update(other.bonds)
        # include bond tags from other network
        if self.btags.keys() & other.btags.keys():
            raise ValueError("keys for the bond tags in the two networks must be unique")
        self.btags.update(other.btags)
        # enable chaining
        return self

    def is_consistent(self, verbose=False):
        """
        Perform an internal consistency check,
        e.g., whether the bond ID specified by any tensor actually exist.
        """
        for k, tensor in self.tensors.items():
            if k != tensor.tid:
                if verbose: print(f"Consistency check failed: dictionary key {k} does not match tensor ID {tensor.tid}.")
                return False
            for ax, bid in enumerate(tensor.bids):
                # bond with ID 'bid' must exist
                if bid not in self.bonds:
                    if verbose: print(f"Consistency check failed: bond with ID {bid} referenced by tensor {k} does not exist.")
                    return False
                # bond must refer back to tensor and corresponding axis
                bond = self.bonds[bid]
                if (tensor.tid, ax) not in zip(bond.tids, bond.axes):
                    if verbose: print(f"Consistency check failed: bond with ID {bid} does not refer to axis {ax} of tensor {tensor.tid}.")
                    return False
        for k, bond in self.bonds.items():
            if k != bond.bid:
                if verbose: print(f"Consistency check failed: dictionary key {k} does not match bond ID {bond.bid}.")
                return False
            dims = []
            for tid, ax in zip(bond.tids, bond.axes):
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
                if not all([d == dims[0] for d in dims]):
                    if verbose: print(f"Consistency check failed: axes dimensions of bond {bond.bid} are not all the same.")
                    return False
        for k, tbids in self.btags.items():
            for bid in tbids:
                if bid not in self.bonds:
                    if verbose: print(f"Consistency check failed: bond with ID {bid} referenced in tag {k} does not exist.")
                    return False
        return True
