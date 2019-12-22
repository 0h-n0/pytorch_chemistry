from pathlib import Path

import torch
import torch.nn.functional as F
import torch_geometric
from rdkit import Chem
from rdkit.Chem import rdmolops

from .base_dataset import BaseDataset
from .utils import (check_download_file_size,
                    check_local_file_size,
                    extract_zipfile,
                    to_sparse,
                    download)
from ..utils import to_Path


class Tox21Dataset(BaseDataset):
    _urls = {
        'train': {
            'url': 'https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf',
            'filename': 'tox21_10k_data_all.sdf.zip'},
        'val': {
            'url': 'https://tripod.nih.gov/tox21/challenge/download?'
            'id=tox21_10k_challenge_testsdf',
            'filename': 'tox21_10k_challenge_test.sdf.zip'
        },
        'test': {
            'url': 'https://tripod.nih.gov/tox21/challenge/download?'
            'id=tox21_10k_challenge_scoresdf',
            'filename': 'tox21_10k_challenge_score.sdf.zip'
        }
    }
    _label_names = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                    'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                    'SR-HSE', 'SR-MMP', 'SR-p53']

    def __init__(self, target='train', return_smiles=False, savedir='.',
                 sparse=False, none_label=-1, max_atoms=0, max_atom_types=0,
                 save_preprocesed_file=True):
        self.target = target
        self.return_smiles = return_smiles
        self.savedir = savedir
        self.sparse = sparse
        self.none_label = none_label
        self.max_atoms = max_atoms
        self.max_atom_types = max_atom_types
        self.save_preprocesed_file = save_preprocesed_file
        zfilename = self._download(target, savedir)
        extracted_files = extract_zipfile(zfilename, savedir)
        filename = extracted_files[0]
        self.mols = self._preprocess(filename)

    def _preprocess(self, filename):
        tmpmols = Chem.SDMolSupplier(filename)
        mols = []
        for m in tmpmols:
            if m is None:
                continue
            try:
                n_atoms = m.GetNumAtoms()
                n_atom_types = max([a.GetAtomicNum() for a in m.GetAtoms()])
                if self.max_atoms < n_atoms:
                    self.max_atoms = n_atoms
                if self.max_atom_types < n_atom_types:
                    self.max_atom_types = n_atom_types
            except Exception as e:
                print(e)
                pass
            mols.append(m)
        return mols

    def _get_label(self, mol: Chem):
        labels = []
        for label in self._label_names:
            if mol.HasProp(label):
                labels.append(int(mol.GetProp(label)))
            else:
                labels.append(self.none_label)
        return torch.tensor(labels)

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        N = self.mols[idx].GetNumAtoms()
        atoms = torch.tensor([a.GetAtomicNum() for a in self.mols[idx].GetAtoms()])
        padded_atoms = torch.zeros(self.max_atoms).long()
        padded_atoms[:N] = atoms
        padded_atoms = torch.eye(self.max_atoms, self.max_atom_types)[padded_atoms, :]
        padded_atoms[:, 0] = 0
        edge = to_sparse(
            torch.from_numpy(rdmolops.GetAdjacencyMatrix(self.mols[idx])),
            self.max_atoms)

        if self.sparse:
            padded_atoms = to_sparse(padded_atoms)
        else:
            edge = edge.to_dense()
        label = self._get_label(self.mols[idx]).long()
        return padded_atoms, edge, label

    def _download(self, target, savedir='.') -> Path:
        url = self._urls[target]['url']
        filename = self._urls[target]['filename']
        savefilename = to_Path(savedir) / filename
        if savefilename.exists():
            local_file_size = check_local_file_size(savefilename)
            download_file_size = check_download_file_size(url)
            if local_file_size != download_file_size:
                download(url, filename, savedir)
        else:
            download(url, filename, savedir)
        return savefilename