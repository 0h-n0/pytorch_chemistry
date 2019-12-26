# pytorch-chemistry

This repo is forked from chainer-chemistry. Please also check [chainer-chemistry](https://github.com/pfnet-research/chainer-chemistry).

The main purpose of this library is providing all of moudules support batch dimension calculation. This means the modules can treat 3-d node features have batch dimension in fisrt position. If you don't need batch dimension, you should use [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric).

Any pull requests are welcome.

### Graph Convolution Implemeted lists

* **[GCNConv](https://github.com/0h-n0/pytorch_chemistry/blob/master/torch_chemistry/nn/conv/gcn_conv.py)** from Kipf and Welling: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (ICLR 2017)
* **[SAGEConv](https://github.com/0h-n0/pytorch_chemistry/blob/master/torch_chemistry/nn/conv/sage_conv.py)** from Hamilton *et al.*: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) (NIPS 2017)
* **[CensNetConv](https://github.com/0h-n0/pytorch_chemistry/blob/master/torch_chemistry/nn/conv/censnet_conv.py)** from Xiaodong Jiang *et al.*: [CensNet: Convolution with Edge-Node Switching in Graph Neural Networks](https://www.ijcai.org/proceedings/2019/0369.pdf)

### Graph Explanabilty Implemeted lists

* WIP

### Graph Generative Implemented lists

* WIP

#### Supported datasets

* tox21
* qm9 (not yet)

### Requirements

* pytorch >= 1.3
* rdkit >= 2018.03.1.0
* torch-geometric


### Dataset Description

* tox21
  * https://tripod.nih.gov/tox21/challenge/
* qm9
  * http://quantum-machine.org/datasets/

### References

* Chainer-chemistry
  * https://github.com/pfnet-research/chainer-chemistry
* DeepChem
  * https://github.com/deepchem/deepchem
