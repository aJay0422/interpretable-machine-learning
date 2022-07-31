import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from model import CNN_CIFAR10
from data import prepare_CIFAR10
from utils import get_loss_acc


test_acc_tracin = []
test_acc_random = []

# prepare data
trainloader, testloader = prepare_CIFAR10(mode="tt")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = CNN_CIFAR10()
net.to(device)

# baseline
net.load_state_dict(torch.load("experiment3/CNN_CIFAR10_baseline_epoch70.pth", map_location=device))
_, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
test_acc_tracin.append(acc)
test_acc_random.append(acc)

for i in range(2, 10):
    # tracin
    net.load_state_dict(torch.load("experiment3/pick small/CNN_CIFAR10_drop{}.pth".format(int(i * 10)), map_location=device))
    _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
    test_acc_tracin.append(acc)
    # random
    net.load_state_dict(torch.load("experiment3/pick small/CNN_CIFAR10_drop{}random.pth".format(int(i * 10)), map_location=device))
    _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
    test_acc_random.append(acc)

xrange = [0] + [0.1 * i for i in range(2, 10)]
plt.plot(xrange, test_acc_tracin, '-bo', label="TracIn")
plt.plot(xrange, test_acc_random, '-ro', label="Random")
plt.title("Pick Small Self-Influence")
plt.legend()
plt.show()
