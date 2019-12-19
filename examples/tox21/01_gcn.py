#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from pytorch_chemistry.datasets.tox21 import Tox21Dataset

batch_size = 128

train_loader = torch.utils.data.DataLoader(Tox21Dataset('train'),
                                           batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(Tox21Dataset('val'),
                                         batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(Tox21Dataset('test'),
                                         batch_size=batch_size, shuffle=True)
