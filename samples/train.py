#!/usr/bin/env python3
import sys
import importlib
from pathlib import Path

import h5py
import torch
import torch.utils.data
from sacred import Experiment
from sacred.observers import FileStorageObserver

from e2edd import DATAPATH
from e2edd import MAX_ATOMIC_NUM
import pytorch_chemistry.models as models
from e2edd.utils.mylogger import mylogger
from e2edd.utils.sacred_helper import load_config
from pytorch_chemistry.data import NumpyTupleDataset
from pytorch_chemistry.data import HDF5Dataset
from pytorch_chemistry.data import collate_fn
from pytorch_chemistry.utils.trainer import Trainer

experiment_name = 'e2edd-train'

ex = Experiment(experiment_name)
ex.logger = mylogger
ex.logger.setLevel('INFO')

@ex.config
def config():
    dataset_type = 'tox21'    
    assert dataset_type in ['tox21', 'qm9'], f'Not support {dataset_type}'
    workdir = (Path('exp') / ('train_' + dataset_type))
    datadir = str(DATAPATH / ('preprocessed_' + dataset_type))
    observer = FileStorageObserver.create(str(Path(workdir).resolve()
                                              / 'config'))
    method = load_config(datadir)['method']
    assert method in models.methods, \
        f'Not implement {method} yet.'
    
    ngpu = 1
    epochs = 100
    log_interval = 10
    benchmark_mode = True
    batch_size = 1
    
    unit_num = 12 # number of units in one layer of the model
    frequency = -1
    eval_mode = 0 
    # '0: only binary_accuracy is calculated.'
    # '1: binary_accuracy and ROC-AUC score is calculate
    assert eval_mode in [0, 1], ('0: only binary_accuracy is calculated.'
                                 '1: binary_accuracy and ROC-AUC score is calculate')
    num_processes_per_iterator = 1

    optimizer_type = 'Adam'
    if optimizer_type == 'Adam':
        optimizer_options = dict(
            lr=0.003,
            weight_decay=0
        )
    else:
        optimizer_options = dict(
            lr=0.001
        )

    dropout = 0.5

    graph_linear_options = dict(
        dropout=0.0,
        init_xavier=True,
        init_normal=True,
        init_gain=1.0,
        )

    rnn_options = dict(
        bidirectional=False, # must be False
        dropout=0.0,
        batch_first=True,
        init_xavier=True,
        init_normal=True,
        init_gain=1.0,
        concat=True,
        )

    if method == 'ggnn':
        model_options = dict(
            out_dim=unit_num,
            hidden_dim=16,
            n_layers=4,
            n_atom_types=MAX_ATOMIC_NUM,
            concat_hidden=False,
            weight_tying=True,
            rnn_type='GRU',
            config_glayer=graph_linear_options,
            config_rnn=rnn_options
            )
    elif method == 'rsgcn':
        model_options = dict(
            out_dim=unit_num,
            hidden_dim=32,
            n_layers=4,
            use_batch_norm=False,
            readout=None
            )
    else:
        raise NotImplementedError(f'Only support {method} Model.')
        
    ex.observers.append(observer)
    del observer

@ex.capture
def get_optimizer(model, optimizer_type, optimizer_options):
    optimizer = getattr(torch.optim, optimizer_type)(model.parameters(),
                                                     **optimizer_options)
    return optimizer
    
@ex.capture
def get_model(method, model_options):
    model = models.method_to_model[method](**model_options)
    return model

@ex.capture
def get_labels(datadir):
    label_names = load_config(datadir)['label_names']
    return label_names

@ex.capture
def get_trainer(model, train_iterator, cv_iterator, test_iterator,
                ngpu, epochs, batch_size, log_interval, benchmark_mode,
                workdir, optimizer_type, optimizer_options
):
    optimizer = get_optimizer(model=model)
    label_names = get_labels()    
    options = dict(
        model=model,
        optimizer=optimizer,
        train_iterator=train_iterator,
        cv_iterator=cv_iterator,
        test_iterator=test_iterator,
        epochs=epochs,
        log_interval=log_interval,
        benchmark_mode=benchmark_mode,
        ngpu=ngpu,
        workdir=workdir,
        label_names=label_names
    )
    t = Trainer(**options)
    return t

@ex.capture
def get_iterators(_log, batch_size, datadir, num_processes_per_iterator):
    datadir = Path(datadir).expanduser().resolve()    
    _log.info(f'datadir {datadir}')
    train_dataset = NumpyTupleDataset.load_hdf5(str(datadir / 'train.hdf5'))
    _log.info(f"load {str(datadir / 'train.hdf5')}")        
    cv_dataset = NumpyTupleDataset.load_hdf5(str(datadir / 'cv.hdf5'))
    _log.info(f"load {str(datadir / 'cv.hdf5')}")    
    test_dataset = NumpyTupleDataset.load_hdf5(str(datadir / 'test.hdf5'))
    _log.info(f"load {str(datadir / 'test.hdf5')}")
    options = dict(
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=num_processes_per_iterator,
        collate_fn=collate_fn
    )
    
    train_dataset = HDF5Dataset(train_dataset)
    train_iterator = torch.utils.data.DataLoader(train_dataset, **options)
    options['shuffle'] = False
    cv_dataset = HDF5Dataset(cv_dataset)
    test_dataset = HDF5Dataset(test_dataset)    
    cv_iterator = torch.utils.data.DataLoader(cv_dataset, **options)
    test_iterator = torch.utils.data.DataLoader(test_dataset, **options)
    return train_iterator, cv_iterator, test_iterator

@ex.capture    
def check_config(_config):
    pass
    
@ex.automain
def main():
    check_config()
    train_iterator, cv_iterator, test_iterator = get_iterators()
    model = get_model()
    trainer = get_trainer(model=model,
                          train_iterator=train_iterator,
                          cv_iterator=cv_iterator,
                          test_iterator=test_iterator)
    trainer.run()
