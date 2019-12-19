import os
import shutil
from pathlib import Path
from zipfile import ZipFile
from typing import List

import requests
import torch

from ..utils import to_Path


def check_download_file_size(url: str) -> int:
    res = requests.head(url)
    size = res.headers['content-length']
    return int(size)

def check_local_file_size(filename: str) -> int:
    p = to_Path(filename)
    info = os.stat(p)
    return info.st_size

def download(url: str = '', filename: str = '', savedir: str ='.') -> int:
    savefile = to_Path(savedir) / filename
    if not savefile.exists():
        with requests.get(url, stream=True) as r:
            with open(savefile, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    return savefile

def extract_zipfile(zfilename: str, extractdir: str ='.') -> List[str]:
    with ZipFile(zfilename) as zipfile:
        zipfile.extractall(extractdir)
        namelist = zipfile.namelist()
    return namelist

def to_sparse(x: torch.tensor, max_size: int = None):
    """ ref: https://discuss.pytorch.org/t/how-to-convert-a-dense-matrix-to-a-sparse-one/7809 """
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)

    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    if max_size is None:
        return sparse_tensortype(indices, values, x.size())
    else:
        return sparse_tensortype(indices, values, (max_size, max_size))
