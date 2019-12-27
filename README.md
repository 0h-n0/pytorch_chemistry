# pytorch-chemistry

This repo is forked from chainer-chemistry. Please also check [chainer-chemistry](https://github.com/pfnet-research/chainer-chemistry).

The main purpose of this library based on [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) is for providing chemistry and biology tasks.

Any pull requests are welcome.

### Graph Convolution Implemeted lists

* **[CensNetConv](https://github.com/0h-n0/pytorch_chemistry/blob/master/torch_chemistry/nn/conv/censnet_conv.py)** from Xiaodong Jiang *et al.*: [CensNet: Convolution with Edge-Node Switching in Graph Neural Networks](https://www.ijcai.org/proceedings/2019/0369.pdf)

### Implemeted lists: Graph Explanabilty

* WIP

###  Implemented lists: Graph Generative

* WIP

#### Supported datasets

* tox21

### Requirements

* pytorch >= 1.3
* pytorch_geometric >= 1.3
* rdkit >= 2019

### Dataset Description

* tox21
  * https://tripod.nih.gov/tox21/challenge/
* qm9
  * http://quantum-machine.org/datasets/

### Related works

* Chainer-chemistry
  * https://github.com/pfnet-research/chainer-chemistry
* DeepChem
  * https://github.com/deepchem/deepchem
