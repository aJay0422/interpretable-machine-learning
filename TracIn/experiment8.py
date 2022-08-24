"""
Reproduce mislabelled data identification
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from utils import mydataset
from model import CNN_CIFAR10
from train import train_CNN_CIFAR10


def prepare_CIFAR10_wrong(mode="tt", seed=42):
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    train_set = torchvision.datasets.CIFAR10(root="datasets/CIFAR10/", train=True,
                                             download=False, transform=data_transform["train"])
    test_set = torchvision.datasets.CIFAR10(root="datasets/CIFAR10/", train=False,
                                            download=False, transform=data_transform["val"])
    X_train = train_set.data.transpose((0, 3, 1, 2))
    X_test = test_set.data.transpose((0, 3, 1, 2))
    Y_train = train_set.targets
    Y_test = test_set.targets

    # assign wrong labels
    n_wrong = int(0.1 * len(Y_train))
    np.random.seed(seed)
    wrong_index = np.random.permutation(len(Y_train))[:n_wrong]
    for index in wrong_index:
        true_label = Y_train[index]
        wrong_labels = [0,1,2,3,4,5,6,7,8,9]
        wrong_labels.remove(true_label)
        Y_train[index] = np.random.choice(wrong_labels)

    train_set = mydataset(X_train, Y_train)
    test_set = mydataset(X_test, Y_test)
    trainloader = DataLoader(train_set, batch_size=256,
                             shuffle=True)
    testloader = DataLoader(test_set, batch_size=64,
                            shuffle=False)
    print(len(X_train), len(X_test))
    return trainloader, testloader



if __name__ == "__main__":
    trainloader, testloader = prepare_CIFAR10_wrong()
    train_CNN_CIFAR10(70, trainloader, testloader,
                      save_path="experiment8/CNN_CIFAR10_wrong.pth")

