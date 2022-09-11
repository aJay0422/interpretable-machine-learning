"""random keep part of the training data"""
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

from utils import prepare_CIFAR10, mydataset
from train import train_CNN_CIFAR10


def main(keep_ratios=[0.1, 0.2, 0.4, 0.6, 0.8]):
    trainloader, valloader, testloader = prepare_CIFAR10()
    n_train = len(trainloader.dataset.Label)
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label
    for keep_ratio in keep_ratios:
        # Prepare data
        n_keep = int(n_train * keep_ratio)
        np.random.seed(42)
        keep_index = np.random.permutation(n_train)[:n_keep]
        X_train_keep = X_train[keep_index]
        Y_train_keep = Y_train[keep_index]
        train_set_keep = mydataset(X_train_keep, Y_train_keep)
        trainloader_keep = DataLoader(train_set_keep, batch_size=128,
                                      shuffle=True)

        # Train model
        save_path = "data/random_keep/random_keep_{}.pth".format(keep_ratio)
        train_CNN_CIFAR10(epochs=150, trainloader=trainloader_keep,
                          testloader=valloader, save_path=save_path)


if __name__ == "__main__":
    main()