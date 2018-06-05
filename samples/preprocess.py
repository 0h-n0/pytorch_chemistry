#!/usr/bin/env python3
from pathlib import Path

from sacred import Experiment
from sacred.observers import FileStorageObserver

from e2edd import DATAPATH
from e2edd.utils.mylogger import mylogger
from pytorch_chemistry.data import NumpyTupleDataset
from pytorch_chemistry.data.preprocessors import preprocess_method_dict
import pytorch_chemistry.data.tox21 as D

experiment_name = 'e2edd-preprocess'

ex = Experiment(experiment_name)
ex.logger = mylogger
ex.logger.setLevel('INFO')

@ex.config
def config():
    dataset_type = 'tox21'
    assert dataset_type in ['tox21', 'qm9']    
    workdir = str(DATAPATH / ('preprocessed_' + dataset_type))
    observer = FileStorageObserver.create(str(Path(workdir).resolve()
                                              / 'config'))
    if dataset_type == 'tox21':
        label_names = D.get_tox21_label_names()
    elif dataset_type == 'qm9':
        label_names = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
                       'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    scale_list = ['standardize', 'none']
    method = 'ggnn' # ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn']
    assert method in ['nfp', 'ggnn', 'schnet', 'weavenet', 'rsgcn'], \
        'NotImplementError'

    outtype = 'hdf5'
    assert outtype in ['npz', 'json', 'hdf5'], f'Not support {outtype} ad outtype.'
    
    
    ex.observers.append(observer)
    del observer

@ex.capture    
def check_config(_config):
    pass
    
@ex.automain
def main(_log, method, workdir, label_names, outtype):
    check_config()
    _log.info(f'label_names = {label_names}')
    workdir = Path(workdir).expanduser().resolve()
    preprocessor = preprocess_method_dict[method]()
    train, val, test = D.get_tox21(preprocessor, labels=label_names)
    # TODO: preprocess
    # TODO: dump json
    if outtype == 'npz':
        _log.info(f"dump {str(workdir / 'train.npz')}")
        NumpyTupleDataset.save(str(workdir / 'train.npz'), train)
        _log.info(f"dump {str(workdir / 'val.npz')}")    
        NumpyTupleDataset.save(str(workdir / 'val.npz'), val)
        _log.info(f"dump {str(workdir / 'test.npz')}")    
        NumpyTupleDataset.save(str(workdir / 'test.npz'), test)
    elif outtype == 'json':
        _log.info(f"dump {str(workdir / 'train.json')}")    
        NumpyTupleDataset.save_json(str(workdir / 'train.json'), train)
        _log.info(f"dump {str(workdir / 'val.json')}")        
        NumpyTupleDataset.save_json(str(workdir / 'val.json'), val)    
        _log.info(f"dump {str(workdir / 'test.json')}")        
        NumpyTupleDataset.save_json(str(workdir / 'test.json'), test)
    elif outtype == 'hdf5':
        _log.info(f"dump {str(workdir / 'train.hdf5')}")    
        NumpyTupleDataset.save_hdf5(str(workdir / 'train.hdf5'), train)
        _log.info(f"dump {str(workdir / 'cv.hdf5')}")        
        NumpyTupleDataset.save_hdf5(str(workdir / 'cv.hdf5'), val)    
        _log.info(f"dump {str(workdir / 'test.hdf5')}")        
        NumpyTupleDataset.save_hdf5(str(workdir / 'test.hdf5'), test)
    
