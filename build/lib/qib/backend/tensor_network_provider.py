from typing import Sequence
import h5py
from qib.backend import QuantumProvider
from qib.circuit import Circuit
from qib.field import Field


class TensorNetworkProvider(QuantumProvider):

    def submit(self, circ: Circuit, fields: Sequence[Field], description):
        """
        Submit a quantum circuit to a backend provider,
        returning a "job" object to query the results.
        """
        net = circ.as_tensornet(fields)
        # store network in a HDF5 file
        with h5py.File(description["filename"], "w") as f:
            tgrp = f.create_group("tensors")
            for tensor in net.net.tensors.values():
                dset = tgrp.create_dataset(str(tensor.tid), data=() if tensor.tid == -1 else net.data[tensor.dataref].astype(complex))
                dset.attrs["tid"] = tensor.tid
                dset.attrs["bids"] = tensor.bids
        job = { "net": net }
        return job


    def query_results(self, job):
        """
        Query results of a previously submitted job.
        """
