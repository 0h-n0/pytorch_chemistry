#!/usr/bin/env python
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchex.nn as exnn
from torch_chemistry.nn.conv.gcn_conv import GCNConv
from torch_chemistry.datasets.tox21 import Tox21Dataset

class GCN(nn.Module):
    def __init__(self, max_atom_types, output_channels):
        super(GCN, self).__init__()
        self.gcn1 = GCNConv(max_atom_types, 200)
        self.gcn2 = GCNConv(200, 150)
        self.dropout1 = nn.Dropout(0.25)
        self.linear1 = exnn.Linear(100)
        self.linear2 = exnn.Linear(output_channels)

    def forward(self, nodes, edges):
        x = self.gcn1(nodes, edges)
        x = F.relu(x)
        x = self.gcn2(x, edges)
        x = F.relu(x)
        x = F.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return F.sigmoid(x)


def one_epoch(args, mode, model, device, loader, optimizer, epoch):
    if mode == 'train':
        model.train()
    else:
        model.eval()
    loss = 0
    correct = 0
    for batch_idx, (nodes, edges, labels) in enumerate(loader):
        nodes, edges, labels = nodes.to(device), edges.to(device), labels.to(device)
        if mode == 'train':
            optimizer.zero_grad()
        mask = labels[labels ==  -1]
        output = model(nodes, edges)
        # todo masked binary cross entropy loss
        loss = F.nll_loss(output, target)
        if mode == 'train':
            loss.backward()
            optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                mode, epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def main():
    parser = argparse.ArgumentParser(description='PyTorch tox21 Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--max_atoms', type=int, default=150, metavar='N',
                        help='set maximum atoms in dataset')
    parser.add_argument('--max_atom_types', type=int, default=100, metavar='N',
                        help='set maximum number of the atom type in dataset')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(Tox21Dataset('train', max_atoms=args.max_atoms,
                                                            max_atom_types=args.max_atom_types),
                                               batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(Tox21Dataset('val', max_atoms=args.max_atoms,
                                                          max_atom_types=args.max_atom_types),
                                             batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(Tox21Dataset('test', max_atoms=args.max_atoms,
                                                           max_atom_types=args.max_atom_types),
                                              batch_size=args.batch_size, shuffle=True)

    model = GCN(args.max_atom_types).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        one_epoch(args, 'train', model, device, train_loader, optimizer, epoch)
        one_epoch(args, 'val', model, device, train_loader, optimizer, epoch)
        scheduler.step()
    one_epoch(args, 'test', model, device, train_loader, optimizer, epoch)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
