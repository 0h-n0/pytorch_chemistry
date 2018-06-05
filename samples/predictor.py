 #!/usr/bin/env python3
from pathlib import Path

from sacred import Experiment
from sacred.observers import FileStorageObserver

from e2edd import DATAPATH
from e2edd.utils import mylogger
import pytorch_chemistry.models

experiment_name = 'e2edd-predictor'

ex = Experiment(experiment_name)
ex.logger = mylogger
ex.logger.setLevel('INFO')

@ex.config
def config():
    model = 'MLP'
    
    assert model in ['GCNN', 'MLP', 'NFP', 'RSGCN', 'SchNet', 'WeaveNet'], \
        f'Not implement {model} yet.'
    assert dataset_type in ['tox21', 'qm9'], f'Not support {dataset_type}'
    workdir = (Path('exp') / 'predict_' + dataset_type)
    observer = FileStorageObserver.create(str(Path(workdir).resolve()
                                              / 'config'))
    ex.observers.append(observer)
    del observe
