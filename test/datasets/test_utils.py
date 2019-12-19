from pathlib import Path

from torch_chemistry.utils import to_Path
from torch_chemistry.datasets.utils import (check_download_file_size,
                                              check_local_file_size,
                                              download)

import pytest


@pytest.fixture
def url():
    return 'https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png'

def test_check_download_file_size(url):
    size = check_download_file_size(url)
    assert size == 13504

def test_download(tmpdir, url):
    filename = 'test.png'
    savefile = download(url, filename, tmpdir)
    assert savefile.exists()

def test_check_local_file_size(tmpdir):
    p = Path(tmpdir / 'tmp.txt')
    with p.open('w') as f:
        f.write('abc')
    size = check_local_file_size(p)
    assert size == 3
