
from pytorch_chemistry.datasets.tox21 import Tox21Dataset



def test_download(tmpdir):
    d = Tox21Dataset()
    d._download('train', tmpdir)
    d[1]
