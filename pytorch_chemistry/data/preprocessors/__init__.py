from pytorch_chemistry.data.preprocessors.atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA
from pytorch_chemistry.data.preprocessors.base_preprocessor import BasePreprocessor  # NOQA
from pytorch_chemistry.data.preprocessors.common import construct_adj_matrix  # NOQA
from pytorch_chemistry.data.preprocessors.common import construct_atomic_number_array  # NOQA
from pytorch_chemistry.data.preprocessors.common import MolFeatureExtractionError  # NOQA
from pytorch_chemistry.data.preprocessors.common import type_check_num_atoms  # NOQA
from pytorch_chemistry.data.preprocessors.ecfp_preprocessor import ECFPPreprocessor  # NOQA
from pytorch_chemistry.data.preprocessors.ggnn_preprocessor import GGNNPreprocessor  # NOQA
from pytorch_chemistry.data.preprocessors.mol_preprocessor import MolPreprocessor  # NOQA
from pytorch_chemistry.data.preprocessors.nfp_preprocessor import NFPPreprocessor  # NOQA
from pytorch_chemistry.data.preprocessors.rsgcn_preprocessor import RSGCNPreprocessor  # NOQA
from pytorch_chemistry.data.preprocessors.schnet_preprocessor import SchNetPreprocessor  # NOQA
from pytorch_chemistry.data.preprocessors.weavenet_preprocessor import WeaveNetPreprocessor  # NOQA

preprocess_method_dict = {
    'ecfp': ECFPPreprocessor,
    'nfp': NFPPreprocessor,
    'ggnn': GGNNPreprocessor,
    'schnet': SchNetPreprocessor,
    'weavenet': WeaveNetPreprocessor,
    'rsgcn': RSGCNPreprocessor,
}
