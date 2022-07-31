import torch
import torch.nn as nn

from model import CNN_CIFAR10
from data import prepare_CIFAR10
from utils import get_loss_acc


settings = ["V1T1", "V2T2", "V1T5", "V2T6"]
test_accs = {}

trainloader, valloader, testloader = prepare_CIFAR10()

for setting in settings:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = CNN_CIFAR10()
    weight_path = "experiment1/{}/CNN_CIFAR10_{}.pth".format(setting, setting)
    net.load_state_dict(torch.load(weight_path, map_location=device))

    _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
    test_accs[setting] = acc

print(test_accs)
