import h5py
import torch
from time import time
from typing import List
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import scatter_sum
from .data import Data, Batch
from ..sp_utils import tensor_idx, has_duplicates, sparse_sample
from ..utils import sp_debug


__all__ = ['NAG', 'NAGBatch']


class NAG:
    """Holder for a Nested Acyclic Graph, containing a list of
    nested partitions of the same point cloud.
    """

    def __init__(self, data_list: List[Data]):
        assert len(data_list) > 0, \
            "The NAG must have at least 1 level of hierarchy. Please " \
            "provide a minimum of 1 Data object."
        self._list = data_list
        if sp_debug.is_debug_enabled():
            self.debug()

    def __iter__(self):
        for i in range(self.num_levels):
            yield self[i]

    def get_sub_size(self, high, low=0):
        """Compute the number of points of level 'low' contained in
        each superpoint of level 'high'.

        Note: 'low=-1' is accepted when level-0 has a 'sub' attribute
        (ie level-0 points are themselves clusters of '-1' level
        absent from the NAG object).
        """
        assert -1 <= low < high < self.num_levels
        assert 0 <= low or self[0].is_super

        # Sizes are computed in a bottom-up fashion. Note this scatter
        # operation assumes all levels of hierarchy use dense,
        # consecutive indices which are consistent between levels
        sub_size = self[low + 1].sub.size
        for i in range(low + 1, high):
            sub_size = scatter_sum(sub_size, self[i].super_index, dim=0)
        return sub_size

    def get_super_index(self, high, low=0):
        """Compute the super_index linking the points at level 'low'
        with superpoints at level 'high'.

        Note: 'low=-1' is accepted when level-0 has a 'sub' attribute
        and 'high=self.num_levels + 1' is accepted when the last level
        has a 'super_index' attribute.
        """
        assert -1 <= low < high <= self.num_levels
        assert 0 <= low or self[0].is_super
        assert high < self.num_levels or self._list[-1].is_sub

        low = -1 if low < 0 else low

        super_index = self[0].sub.to_super_index() if low < 0 \
            else self[low].super_index

        for i in range(low + 1, high):
            super_index = self[i].super_index[super_index]

        return super_index

    @property
    def num_levels(self):
        """Number of levels of hierarchy in the nested graph."""
        return len(self._list)

    @property
    def num_points(self):
        """Number of points/nodes in the lower-level graph."""
        return [d.num_points for d in self] if self.num_levels > 0 else 0

    def to_list(self):
        """Return the Data list"""
        return self._list

    def clone(self):
        """Return a new NAG instance containing the Data clones."""
        return self.__class__([d.clone() for d in self])

    def detach(self):
        """Detach all tensors in the NAG."""
        self._list = [d.detach() for d in self]
        return self

    def to(self, device, **kwargs):
        """Move the NAG with all Data in it to device."""
        self._list = [d.to(device, **kwargs) for d in self]
        return self

    def cpu(self, **kwargs):
        """Move the NAG with all Data in it to CPU."""
        return self.to('cpu', **kwargs)

    def cuda(self, **kwargs):
        """Move the NAG with all Data in it to CUDA."""
        return self.to('cuda', **kwargs)

    @property
    def device(self):
        """Return device of first Data in NAG."""
        return self[0].device if self.num_levels > 0 \
            else torch.tensor([]).device

    @property
    def is_cuda(self):
        """Return True is one of the Data contains a CUDA Tensor."""
        for d in self:
            if isinstance(d, torch.Tensor) and d.is_cuda:
                return True
        return False

    def __getitem__(self, idx):
        """Return a Data object from the hierarchy.

        Parameters
        ----------
        idx: int, slice
            The hierarchy level to return
        """
        if isinstance(idx, int):
            return self._list[idx]
        return NAG(self._list[idx])

    def select(self, i_level, idx):
        """Indexing mechanism on the NAG.

        Returns a new copy of the indexed NAG, with updated clusters.
        Supports int, torch and numpy indexing.

        Contrary to indexing 'Data' objects in isolation, this will
        maintain cluster indices compatibility across all levels of the
        hierarchy.

        Note that cluster indices in 'idx' must be duplicate-free.
        Indeed, duplicates would create ambiguous situations or lower
        and higher hierarchy level updates.

        Parameters
        ----------
        i_level: int
            The hierarchy level to index from.
        idx: int, np.NDArray, torch.Tensor
            Index to select nodes of the chosen hierarchy. Must be
            duplicate-free
        """
        assert isinstance(i_level, int)
        assert i_level < self.num_levels

        # Convert idx to a Tensor
        idx = tensor_idx(idx).to(self.device)

        # Make sure idx contains no duplicate entries
        if sp_debug.is_debug_enabled():
            assert not has_duplicates(idx), \
                "Duplicate indices are not supported. This would cause " \
                "ambiguities in edges and super- and sub- indices."

        # Prepare the output Data list
        data_list = [None] * self.num_levels

        # Select the nodes at level 'i_level' and update edges, subpoint
        # and superpoint indices accordingly. The returned 'out_sub' and
        # 'out_super' will help us update the lower and higher hierarchy
        # levels iteratively
        data_list[i_level], out_sub, out_super = self[i_level].select(
            idx, update_sub=True, update_super=True)

        # Iteratively update lower hierarchy levels
        for i in range(i_level - 1, -1, -1):
            # Unpack the 'out_sub' from the previous above level
            (idx_sub, sub_super) = out_sub

            # Select points but do not update 'super_index', it will be
            # directly provided by the above-level's 'sub_super'
            data_list[i], out_sub, _ = self[i].select(
                idx_sub, update_sub=True, update_super=False)

            # Directly update the 'super_index' using 'sub_super' from
            # the above level
            data_list[i].super_index = sub_super

        # Iteratively update higher hierarchy levels
        for i in range(i_level + 1, self.num_levels):
            # Unpack the 'out_super' from the previous below level
            (idx_super, super_sub) = out_super

            # Select points but do not update 'sub', it will be directly
            # provided by the above-level's 'super_sub'
            data_list[i], _, out_super = self[i].select(
                idx_super, update_sub=False, update_super=True)

            # Directly update the 'sub' using 'super_sub' from the above
            # level
            data_list[i].sub = super_sub

        # Create a new NAG with the list of indexed Data
        nag = self.__class__(data_list)

        return nag

    def save(
            self,
            path,
            y_to_csr=True,
            pos_dtype=torch.float,
            fp_dtype=torch.float):
        """Save NAG to HDF5 file.

        :param path:
        :param y_to_csr: bool
            Convert 'y' to CSR format before saving. Only applies if
            'y' is a 2D histogram
        :param pos_dtype: torch dtype
            Data type to which 'pos' should be cast before saving. The
            reason for this separate treatment of 'pos' is that global
            coordinates may be too large and casting to 'fp_dtype' may
            result in hurtful precision loss
        :param fp_dtype: torch dtype
            Data type to which floating point tensors should be cast
            before saving
        """
        with h5py.File(path, 'w') as f:
            for i_level, data in enumerate(self):
                g = f.create_group(f'partition_{i_level}')
                data.save(
                    g,
                    y_to_csr=y_to_csr,
                    pos_dtype=pos_dtype,
                    fp_dtype=fp_dtype)

    @staticmethod
    def load(
            path,
            low=0,
            high=-1,
            idx=None,
            keys_idx=None,
            keys_low=None,
            keys=None,
            update_super=True,
            update_sub=True,
            verbose=False):
        """Load NAG from an HDF5 file. See `NAG.save` for writing such
        file. Options allow reading only part of the data.

        :param path: str
            Path the file
        :param low: int
            Lowest partition level to read
        :param high: int
            Highest partition level to read
        :param idx: list, array, tensor, slice
            Index or boolean mask used to select from low
        :param keys_idx: list(str)
            Keys on which the indexing should be applied
        :param keys_low: list(str)
            Keys to read for low-level. If None, all keys will be read
        :param keys: list(str)
            Keys to read. If None, all keys will be read
        :param update_sub: bool
            See NAG.select and Data.select
        :param update_super:
            See NAG.select and Data.select
        :param verbose: bool
        :return:
        """
        keys_low = keys if keys_low is None and keys is not None else keys_low

        data_list = []
        with h5py.File(path, 'r') as f:

            # Initialize partition levels min and max to read from the
            # file. This functionality is especially intended for
            # loading levels 1 and above when we want to avoid loading
            # the memory-costly level-0 points
            low = max(low, 0)
            high = len(f) - 1 if high < 0 else min(high, len(f) - 1)

            # Make sure all required partitions are present in the file
            assert all([
                f'partition_{k}' in f.keys()
                for k in range(low, high + 1)])

            # Apply index selection on the low only, if required. For
            # all subsequent levels, only keys selection is available
            for i in range(low, high + 1):
                start = time()
                if i == low:
                    data = Data.load(
                        f[f'partition_{i}'], idx=idx, keys_idx=keys_idx,
                        keys=keys_low, update_sub=update_sub,
                        verbose=verbose)
                else:
                    data = Data.load(
                        f[f'partition_{i}'], keys=keys, update_sub=False,
                        verbose=verbose)
                data_list.append(data)
                if verbose:
                    print(f'NAG.load lvl-{i:<12} : 'f'{time() - start:0.3f}s\n')

        # In the case where update_super is not required but the low
        # level was indexed, we cannot combine the leve-0 and level-1+
        # Data into a NAG, because the indexing might have broken index
        # consistency between the levels. So we return the elements in a
        # NAG.cat_select-friendly way, for later update
        if not update_super and idx is not None:
            return data_list[0], data_list[1:], idx

        # In case the lowest level was indexed, we need to update the
        # above level too. Unfortunately, this is probably because we do
        # not want to load the whole low-level partition, so we
        # artificially create a Data object to simulate it, just to be
        # able to leverage the convenient NAG.select method.
        # NB: this may be a little slow for the CPU-based DataLoader
        # operations at train time, so we will prefer setting
        # update_super=False in this situation and do the necessary
        # later on GPU
        if update_super:
            return NAG.cat_select(data_list[0], data_list[1:], idx=idx)

        return NAG(data_list)

    @staticmethod
    def cat_select(data, data_list, idx=None):
        """Does part of what Data.select does but in an ugly way. This
        is mostly intended for the DataLoader to be able to load NAG and
        sample level-0 points on CPU in reasonable time and finish the
        update_sub, update_super work on GPU later on if need be...

        :param data: Data object for level-0 points
        :param data_list: list of Data objects for level-1+ points
        :param idx: optional, indexing that has been applied on level-0
            data and guides higher levels updating (see NAG.select and
            Data.select with update_super=True)
        :return:
        """
        assert isinstance(data, Data)
        assert isinstance(data_list, list)

        if idx is None and data_list is None or len(data_list) == 0:
            return NAG([data])

        if idx is None:
            return NAG([data] + data_list)

        if data_list is None or len(data_list) == 0:
            data.super_index = consecutive_cluster(data.super_index)[0]
            return NAG([data])

        fake_super_index = data_list[0].sub.to_super_index()
        fake_x = torch.empty_like(fake_super_index)
        data_fake = Data(x=fake_x, super_index=fake_super_index)
        nag = NAG([data_fake] + data_list)
        nag = nag.select(0, idx)
        data.super_index = nag[0].super_index
        nag._list[0] = data

        return nag

    def debug(self):
        """Sanity checks."""
        assert self.num_levels > 0
        for i, d in enumerate(self):
            assert isinstance(d, Data)
            if i > 0:
                assert d.is_super
                assert d.num_points == self[i - 1].num_super
            if i < self.num_levels - 1:
                assert d.is_sub
                assert d.num_points == self[i + 1].num_sub
            d.debug()

    def get_sampling(
            self,
            high=1,
            low=0,
            n_max=32,
            n_min=1,
            mask=None,
            return_pointers=False):
        """Compute indices to sample elements at `low`-level, based on
        which segment they belong to at `high`-level.

        The sampling operation is run without replacement and each
        segment is sampled at least `n_min` and at most `n_max` times,
        within the limits allowed by its actual size.

        Optionally, a `mask` can be passed to filter out some
        `low`-level points.

        :param high: int
            Partition level of the segments we want to sample. By
            default, `high=1` to sample the level-1 segments
        :param low: int
            Partition level we will sample from, guided by the `high`
            segments. By default, `high=0` to sample the level-0 points.
            `low=-1` is accepted when level-0 has a `sub` attribute (ie
            level-0 points are themselves segments of `-1` level absent
            from the NAG object).
        :param n_max: int
            Maximum number of `low`-level elements to sample in each
            `high`-level segment
        :param n_min: int
            Minimum number of `low`-level elements to sample in each
            `high`-level segment, within the limits of its size (ie no
            oversampling)
        :param mask: list, np.ndarray, torch.Tensor
            Indicates a subset of `low`-level elements to consider. This
            allows ignoring
        :param return_pointers: bool
            Whether pointers should be returned along with sampling
            indices. These indicate which of the output `low`-level
            sampling indices belong to which `high`-level segment
        """
        super_index = self.get_super_index(high, low=low)
        return sparse_sample(
            super_index, n_max=n_max, n_min=n_min, mask=mask,
            return_pointers=return_pointers)

    def __repr__(self):
        info = [
            f"{key}={getattr(self, key)}"
            for key in ['num_levels', 'num_points', 'device']]
        return f"{self.__class__.__name__}({', '.join(info)})"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            if sp_debug.is_debug_enabled():
                print(f'{self.__class__.__name__}.__eq__: classes differ')
            return False
        if self.num_levels != other.num_levels:
            if sp_debug.is_debug_enabled():
                print(f'{self.__class__.__name__}.__eq__: num_levels differ')
            return False
        for d1, d2 in zip(self, other):
            if d1 != d2:
                if sp_debug.is_debug_enabled():
                    print(f'{self.__class__.__name__}.__eq__: data differ')
                return False
        return True


class NAGBatch(NAG):
    """Wrapper for NAG batching."""

    @staticmethod
    def from_nag_list(nag_list):
        assert isinstance(nag_list, list)
        assert len(nag_list) > 0
        assert all(isinstance(x, NAG) for x in nag_list)
        return NAGBatch([
            Batch.from_data_list(l) for l in zip(*[n._list for n in nag_list])])

    def to_nag_list(self):
        return [NAG(l) for l in zip(*[b.to_data_list() for b in self])]
