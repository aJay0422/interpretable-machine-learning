import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np

from model import CNN_CIFAR10
from data import prepare_CIFAR10
from train import train_CNN_CIFAR10
from utils import get_loss_acc, mydataset
from influence_functions import tracin_self

if __name__ == "__main__":
    experiment_path = "./experiment2"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    # prepare data
    trainloader, testloader = prepare_CIFAR10(mode="tt")
    n_train = len(trainloader.dataset.Label)

    # # train a baseline model
    # epochs = 50
    # save_path = experiment_path + "/CNN_CIFAR10_basline.pth"
    # summary_baseline = train_CNN_CIFAR10(epochs, trainloader, testloader, save_path=save_path)
    # np.save(experiment_path + "/summary_baseline.npy", summary_baseline)

    # experiment2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = CNN_CIFAR10()
    net.to(device)
    epochs = 50
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    summary = {"train loss": [],
               "test loss": [],
               "train acc": [],
               "test acc": []}
    best_test_acc = 0

    self_influence = np.zeros(n_train)
    for epoch in range(1, epochs+1):
        net.train()
        for X_batch, Y_batch in trainloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # forward
            logits = net(X_batch)
            loss = loss_function(logits, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        with torch.no_grad():
            net.eval()
            # evaluate train
            train_loss, train_acc = get_loss_acc(net, trainloader, loss_function)
            # evaluate test
            test_loss, test_acc = get_loss_acc(net, testloader, loss_function)

        print("Epoch {}/{} train loss: {}  test loss: {}  train acc: {}  test acc: {}".format(epoch, epochs,
                                                                                              train_loss, test_loss,
                                                                                              train_acc, test_acc))
        summary["train loss"].append(train_loss)
        summary["test loss"].append(test_loss)
        summary["train acc"].append(train_acc)
        summary["test acc"].append(test_acc)

        # save model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(net.state_dict(), experiment_path + "/CNN_CIFAR10_exp.pth")

        # update self influence
        X_train = trainloader.dataset.Data
        Y_train = trainloader.dataset.Label
        self_influence_tmp = tracin_self(net, X_train, Y_train)
        self_influence += self_influence_tmp

        # update trainloader
        if epoch % 5 == 0:
            sorted_index = np.argsort(self_influence)[::-1]   # in decreasing order
            X_train_new = X_train[list(sorted_index[:-2500])]
            Y_train_new = Y_train[list(sorted_index[:-2500])]
            trainloader = DataLoader(mydataset(X_train_new, Y_train_new), batch_size=128, shuffle=True)
            n_train = len(Y_train_new)
            self_influence = np.zeros(n_train)
            print("New dataset has {} samples".format(n_train))

    np.save(experiment_path + "/summary.npy", summary)




