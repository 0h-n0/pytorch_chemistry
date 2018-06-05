import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from pytorch_chemistry.models import ggnn  # NOQA
from pytorch_chemistry.models import mlp  # NOQA
from pytorch_chemistry.models import nfp  # NOQA
from pytorch_chemistry.models import schnet  # NOQA
from pytorch_chemistry.models import weavenet  # NOQA

from pytorch_chemistry.models.ggnn import GGNN  # NOQA
from pytorch_chemistry.models.mlp import MLP  # NOQA
from pytorch_chemistry.models.nfp import NFP  # NOQA
from pytorch_chemistry.models.rsgcn import RSGCN  # NOQA
from pytorch_chemistry.models.schnet import SchNet  # NOQA
from pytorch_chemistry.models.weavenet import WeaveNet  # NOQA

methods = ['ggnn', 'mlp', 'nfp', 'rsgcn', 'schnet', 'weavenet']

method_to_model = dict(
    ggnn=GGNN,
    mlp=MLP,
    rsgcn=RSGCN,
    schnet=SchNet,
    weavenet=WeaveNet
    )

class SigmoidCrossEntropyLoss(object):
    def __init__(self, weight=None, size_average=True, reduce=True):
        self.sigmoid = nn.Sigmoid()
        self.cel = nn.BCEWithLogitsLoss(weight=weight,
                                        size_average=size_average
                                        )
        
    def __call__(self, input, target):
        x = self.sigmoid(input)
        return self.cel(x , target)

    def cuda(self):
        self.sigmoid.cuda()
        self.cel.cuda()

        
class Accuracy(object):
    def __init__(self, threshold=0.5, multitask=True):
        self.threshold = threshold

    def __call__(self, input, target):
        batch_size = input.size(0)        
        sigmoid_input = F.sigmoid(input)
        preditions = sigmoid_input.ge(self.threshold).type_as(target)
        correct = preditions.eq(target)
        correct = correct.sum(dim=0)
        correct = 100. * correct.float() / batch_size
        return correct

class AucRocAccuracy(object):
    def __init__(self, threshold=0.5, multitask=True):
        self.threshold = threshold

    def __call__(self, input, target):
        sigmoid_input = F.sigmoid(input)
        r = roc_auc_score(target.data.numpy().tolist(),
                          sigmoid_input.data.numpy().tolist())
        return correct

    def reset_state(self):
        pass
    

        
        
