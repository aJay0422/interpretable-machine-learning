import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from influence_functions import select_train, select_validation, tracin_multi_multi
from data import prepare_CIFAR10
from model import CNN_CIFAR10
from utils import get_loss_acc, mydataset


if __name__ == "__main__":
    net = CNN_CIFAR10()
    trainloader, valloader, testloader = prepare_CIFAR10()
    selected_val = select_validation(net, valloader)

    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label
    n_train = len(Y_train)

    selected_val_X_all = selected_val["X"]
    selected_val_Y_all = selected_val["Y"]
    n_test = len(selected_val_Y_all)

    # compute TracIn
    # TracIn_all = np.load("datasets/selected_cifar10/TracIn_all.npz", allow_pickle=True)["TracIn_all"]
    TracIn_all = tracin_multi_multi(net, X_train, selected_val_X_all, Y_train, selected_val_Y_all)



    stop = None