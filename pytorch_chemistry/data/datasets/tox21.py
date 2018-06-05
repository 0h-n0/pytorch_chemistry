import os
import shutil
import zipfile
from logging import getLogger

import numpy

from pytorch_chemistry.data import download
from pytorch_chemistry.data import NumpyTupleDataset
from pytorch_chemistry.data.preprocessors import preprocess_method_dict
from pytorch_chemistry.data.parsers.sdf_file_parser import SDFFileParser
from pytorch_chemistry.data.preprocessors.atomic_number_preprocessor \
    import AtomicNumberPreprocessor  # NOQA


_config = {
    'train': {
        'url': 'https://tripod.nih.gov/tox21/challenge/download?'
        'id=tox21_10k_data_allsdf',
        'filename': 'tox21_10k_data_all.sdf'
    },
    'val': {
        'url': 'https://tripod.nih.gov/tox21/challenge/download?'
        'id=tox21_10k_challenge_testsdf',
        'filename': 'tox21_10k_challenge_test.sdf'
    },
    'test': {
        'url': 'https://tripod.nih.gov/tox21/challenge/download?'
        'id=tox21_10k_challenge_scoresdf',
        'filename': 'tox21_10k_challenge_score.sdf'
    }
}

_root = 'tox21'

_label_names = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                'SR-HSE', 'SR-MMP', 'SR-p53']


def get_tox21_label_names():
    """Returns label names of Tox21 datasets."""
    return _label_names


def get_tox21(preprocessor=None, labels=None, return_smiles=False):
    """Downloads, caches and preprocesses Tox21 dataset.
    Args:
        preprocesssor (BasePreprocessor): Preprocessor.
            This should be chosen based on the network to be trained.
            If it is None, default `AtomicNumberPreprocessor` is used.
        labels (str or list): List of target labels.
        return_smiles (bool): If set to True, smiles array is also returned.
    Returns:
        The 3-tuple consisting of train, validation and test
        datasets, respectively. Each dataset is composed of `features`,
        which depends on `preprocess_method`.
    """
    labels = labels or get_tox21_label_names()
    if isinstance(labels, str):
        labels = [labels, ]

    def postprocess_label(label_list):
        # Set -1 to the place where the label is not found,
        # this corresponds to not calculate loss with `sigmoid_cross_entropy`
        t = numpy.array([-1 if label is None else label for label in
                         label_list], dtype=numpy.int32)
        return t

    if preprocessor is None:
        preprocessor = AtomicNumberPreprocessor()
    parser = SDFFileParser(preprocessor,
                           postprocess_label=postprocess_label,
                           labels=labels)

    train_result = parser.parse(get_tox21_filepath('train'),
                                return_smiles=return_smiles)
    val_result = parser.parse(get_tox21_filepath('val'),
                              return_smiles=return_smiles)
    test_result = parser.parse(get_tox21_filepath('test'),
                               return_smiles=return_smiles)

    if return_smiles:
        train, train_smiles = train_result['dataset'], train_result['smiles']
        val, val_smiles = val_result['dataset'], val_result['smiles']
        test, test_smiles = test_result['dataset'], test_result['smiles']
        return train, val, test, train_smiles, val_smiles, test_smiles
    else:
        train = train_result['dataset']
        val = val_result['dataset']
        test = test_result['dataset']
        return train, val, test


def _get_tox21_filepath(dataset_type):
    """Returns a filepath in which the tox21 dataset is cached.
    Thie function returns a filepath in which `dataset_type`
    of the tox21 dataset is cached.
    Not that this function does not check if the dataset actually
    has been downloaded or not.
    Args:
        dataset_type(str): Name of the target dataset type.
            Either 'train', 'val', or 'test'.
    Returns (str): filepath for the tox21 dataset
    """
    c = _config[dataset_type]
    sdffile = c['filename']

    cache_root = download.get_dataset_directory(_root)
    cache_path = os.path.join(cache_root, sdffile)
    return cache_path


def get_tox21_filepath(dataset_type, download_if_not_exist=True):
    """Returns a filepath in which the tox21 dataset is cached.
    Thie function returns a filepath in which `dataset_type`
    of the tox21 dataset is or will be cached.
    If the dataset is not cached and if ``download_if_not_exist``
    is ``True``, this function also downloads the dataset.
    Args:
        dataset_type: Name of the target dataset type.
            Either 'train', 'val', or 'test'
    Returns (str): filepath for tox21 dataset
    """
    cache_filepath = _get_tox21_filepath(dataset_type)
    if not os.path.exists(cache_filepath):
        if download_if_not_exist:
            is_successful = _download_and_extract_tox21(dataset_type,
                                                        cache_filepath)
            if not is_successful:
                logger = getLogger(__name__)
                logger.warning('Download failed.')
    return cache_filepath


def _download_and_extract_tox21(config_name, save_filepath):
    is_successful = False
    c = _config[config_name]
    url = c['url']
    sdffile = c['filename']

    # Download tox21 dataset
    download_file_path = download.cached_download(url)

    # Extract zipfile to get sdffile
    with zipfile.ZipFile(download_file_path, 'r') as z:
        z.extract(sdffile)
        shutil.move(sdffile, save_filepath)

    is_successful = True
    return is_successful


def download_and_extract_tox21():
    """Downloads and extracts Tox21 dataset.
    Returns: None
    """
    for config in ['train', 'val', 'test']:
        _download_and_extract_tox21(config, _get_tox21_filepath(config))


class _CacheNamePolicy(object):

    train_file_name = 'train.npz'
    val_file_name = 'val.npz'
    test_file_name = 'test.npz'

    def _get_cache_directory_path(self, method, labels, prefix):
        if labels:
            return os.path.join(prefix, '{}_{}'.format(method, labels))
        else:
            return os.path.join(prefix, '{}_all'.format(method))

    def __init__(self, method, labels, prefix='input'):
        self.method = method
        self.labels = labels
        self.prefix = prefix
        self.cache_dir = self._get_cache_directory_path(method, labels, prefix)

    def get_train_file_path(self):
        return os.path.join(self.cache_dir, self.train_file_name)

    def get_val_file_path(self):
        return os.path.join(self.cache_dir, self.val_file_name)

    def get_test_file_path(self):
        return os.path.join(self.cache_dir, self.test_file_name)

    def create_cache_directory(self):
        try:
            os.makedirs(self.cache_dir)
        except OSError:
            if not os.path.isdir(self.cache_dir):
                raise


def load_dataset(method, labels, prefix='input'):
    policy = _CacheNamePolicy(method, labels, prefix)
    train_path = policy.get_train_file_path()
    val_path = policy.get_val_file_path()
    test_path = policy.get_test_file_path()

    train, val, test = None, None, None
    print()
    if os.path.exists(policy.cache_dir):
        print('load from cache {}'.format(policy.cache_dir))
        train = NumpyTupleDataset.load(train_path)
        val = NumpyTupleDataset.load(val_path)
        test = NumpyTupleDataset.load(test_path)
    if train is None or val is None or test is None:
        print('preprocessing dataset...')
        preprocessor = preprocess_method_dict[method]()
        train, val, test = get_tox21(preprocessor, labels=labels)
        # Cache dataset
        policy.create_cache_directory()
        NumpyTupleDataset.save(train_path, train)
        NumpyTupleDataset.save(val_path, val)
        NumpyTupleDataset.save(test_path, test)
    return train, val, test        