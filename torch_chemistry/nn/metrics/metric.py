import torch.nn as nn

from ..functional.metric import *


class ROCCurve(nn.Module):
    def __init__(self):
        super(ROCCurve, self).__init__()

    def forward(self, pred, target):
        return roc_curve(pred, target)


class ROCAUCScore(nn.Modele):
    def __init__(self):
        super(ROCAUCScore, self).__init__()

    def forward(self, pred, target):
        return roc_auc_score(pred, target)


class AUC(nn.Module):
    def __init__(self):
        super(AUC, self).__init__()

    def forward(self, fpr, tpr):
        return auc(fpr, tpr)
