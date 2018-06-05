import os
import re
import six
import json
import collections

import h5py
import numpy
import torch
from torch._six import string_classes, int_classes
from tqdm import tqdm


from pytorch_chemistry.data.indexers import NumpyTupleDatasetFeatureIndexer  # NOQA


class NumpyTupleDataset(object):
    """Dataset of a tuple of datasets.
    It combines multiple datasets into one dataset. Each example is represented
    by a tuple whose ``i``-th item corresponds to the i-th dataset.
    And each ``i``-th dataset is expected to be an instance of numpy.ndarray.
    Args:
        datasets: Underlying datasets. The ``i``-th one is used for the
            ``i``-th item of each example. All datasets must have the same
            length.
    """

    def __init__(self, *datasets):
        if not datasets:
            raise ValueError('no datasets are given')
        length = len(datasets[0])
        for i, dataset in enumerate(datasets):
            if len(dataset) != length:
                raise ValueError(
                    'dataset of the index {} has a wrong length'.format(i))
        self._datasets = datasets
        self._length = length
        self._features_indexer = NumpyTupleDatasetFeatureIndexer(self)

    def __getitem__(self, index):
        return self._datasets[index]

    def __len__(self):
        return self._length

    def get_datasets(self):
        return self._datasets

    @property
    def features(self):
        """Extract features according to the specified index.
        - axis 0 is used to specify dataset id (`i`-th dataset)
        - axis 1 is used to specify feature index
        .. admonition:: Example
           >>> import numpy
           >>> from chainer_chemistry.datasets import NumpyTupleDataset
           >>> x = numpy.array([0, 1, 2], dtype=numpy.float32)
           >>> t = x * x
           >>> numpy_tuple_dataset = NumpyTupleDataset(x, t)
           >>> targets = numpy_tuple_dataset.features[:, 1]
           >>> print('targets', targets)  # We can extract only target value
           targets [0, 1, 4]
        """
        return self._features_indexer

    @classmethod
    def save(cls, filepath, numpy_tuple_dataset):
        """save the dataset to filepath in npz format
        Args:
            filepath (str): filepath to save dataset. It is recommended to end
                with '.npz' extension.
            numpy_tuple_dataset (NumpyTupleDataset): dataset instance
        """
        if not isinstance(numpy_tuple_dataset, NumpyTupleDataset):
            raise TypeError('numpy_tuple_dataset is not instance of '
                            'NumpyTupleDataset, got {}'
                            .format(type(numpy_tuple_dataset)))
        numpy.savez(filepath, *numpy_tuple_dataset._datasets)

    @classmethod
    def save_json(cls, filepath, numpy_tuple_dataset):
        if not isinstance(numpy_tuple_dataset, NumpyTupleDataset):
            raise TypeError('numpy_tuple_dataset is not instance of '
                            'NumpyTupleDataset, got {}'
                            .format(type(numpy_tuple_dataset)))
        _output = []
        lables = ['atoms', 'adj_matrix', 'assey']
        for i, data in enumerate(numpy_tuple_dataset):
            _length = len(data[0])
            for j in enumerate(data):
                idx = int(j[0])
                if idx == len(_output):
                    _output.append(dict(
                        atoms=j[1].tolist()
                    ))
                else:
                    num_keys = len(_output[idx])
                    if num_keys == 1:
                        _output[idx]['adj_matrix'] = j[1].tolist()                        
                    elif num_keys == 2:
                        _output[idx]['assey'] = j[1].tolist()                                                
                    else:
                        raise NotImplementError(f'Only support {labels} keys')
                        
        with open(filepath, 'w') as fp:
            size = len(_output)
            
    @classmethod
    def save_hdf5(cls, filepath, numpy_tuple_dataset):
        if not isinstance(numpy_tuple_dataset, NumpyTupleDataset):
            raise TypeError('numpy_tuple_dataset is not instance of '
                            'NumpyTupleDataset, got {}'
                            .format(type(numpy_tuple_dataset)))
        _output = []
        lables = ['atoms', 'adj_matrix', 'assey']
        f = h5py.File(filepath, 'w')
        size = len(numpy_tuple_dataset)
        for i, data in tqdm(enumerate(numpy_tuple_dataset),
                            total=size, ascii=True):
            _length = len(data[0])
            for j in enumerate(data):
                idx = j[0]
                if idx == len(_output):
                    f.create_group(f"{idx}")
                    d = f.create_dataset(f'/{idx}/atoms', data=j[1])
                    assert numpy.any(j[1]), (f"this atoms data are almost all 0, idx "
                                             f"idx{idx} ¥n{j[1]}")
                    _output.append(dict(
                        atoms=0
                    ))
                else:
                    num_keys = len(_output[idx])
                    if num_keys == 1:
                        d = f.create_dataset(f'/{idx}/adj_matrix', data=j[1])
                        _output[int(idx)]['adj_matrix'] = 0
                    elif num_keys == 2:
                        d = f.create_dataset(f'/{idx}/assey', data=j[1])
                        _output[int(idx)]['assey'] = 0                        
                    else:
                        raise NotImplementedError(f'Only support {num_keys} keys')
        f.close()
        data =  h5py.File(filepath)                            
        
    @classmethod
    def load_hdf5(cls, filepath):
        if not os.path.exists(filepath):
            return None
        data =  h5py.File(filepath)
        return data
        
    @classmethod
    def load_json(cls, filepath):
        if not os.path.exists(filepath):
            return None
        with open(filepath) as fp:        
            load_data = json.load(fp)
        result = []
        i = 0
        return load_data

    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            return None
        load_data = numpy.load(filepath)
        result = []
        i = 0
        while True:
            key = 'arr_{}'.format(i)
            if key in load_data.keys():
                result.append(load_data[key])
                i += 1
            else:
                break
        return NumpyTupleDataset(*result)

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

_use_shared_memory = False

class SimpleDataset(object):
    def __init__(self, dataset):
        self._dataset = dataset
        self._size = len(self._dataset)

    def __len__(self):
        return self._size

    def __getitem__(self, i):
        return self._dataset[i]

class HDF5Dataset(object):
    def __init__(self, dataset):
        self._dataset = dataset
        self._size = len(self._dataset)
        self.keys = list(dataset['0'].keys())
        
    def __len__(self):
        return self._size

    def __getitem__(self, i):
        _out = {}
        for key in self.keys:
            _out[key] = self._dataset[f'{i}'][key]
        return _out
    
    

def collate_fn(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
            padding = True
            if padding:
                # -- this padding is not sorted --
                # This is because it will be occured to affect training loss
                # with multi gpu training.
                padded_shape = list(batch[0].shape)
                assert len(padded_shape) > 0, "the ditmenssion of padded_shape is smoething wrong."
                inputs = []
                for b in batch[1:]:
                    for idx, (p, _b) in enumerate(zip(padded_shape, b.shape)):
                        if p < _b:
                            padded_shape[idx] = _b
                for i, b in enumerate(batch):
                    b_zeros = torch.zeros(*padded_shape).type(numpy_type_map[b.dtype.name])
                    shapes = b.shape
                    if len(padded_shape) == 1:
                        T = shapes[0]
                        b_zeros[:T] += numpy_type_map[b.dtype.name](b)
                    if len(padded_shape) == 2:
                        T = shapes[0]
                        b_zeros[:T, :T] += numpy_type_map[b.dtype.name](b)
                    elif len(padded_shape) == 3:
                        T = shapes[1]
                        b_zeros[:, :T, :T] += numpy_type_map[b.dtype.name](b)
                    inputs.append(b_zeros)
                return torch.stack(inputs, dim=0)
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        out_dict = {}        
        for key in ['atoms', 'adj_matrix', 'assey']:
            if key == 'atoms':
                out = collate_fn([numpy.array(d[key]) for d in batch])                
                out_dict[key] = torch.stack(out, dim=0)
            elif key == 'adj_matrix':
                out = collate_fn([numpy.array(d[key]) for d in batch])
                out_dict[key] = torch.stack(out, dim=0)                
            elif key == 'assey':
                out = [torch.from_numpy(numpy.array(d[key])) for d in batch]
                out_dict[key] = torch.stack(out, dim=0)                                
            else:
                return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
        return out_dict
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]
    raise TypeError((error_msg.format(type(batch[0]))))

