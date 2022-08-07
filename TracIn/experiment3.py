import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt

from model import CNN_CIFAR10
from data import prepare_CIFAR10
from train import train_CNN_CIFAR10
from utils import get_loss_acc, mydataset
from influence_functions import tracin_self


def baseline():
    # prepare data
    trainloader, testloader = prepare_CIFAR10(mode="tt")
    n_train = len(trainloader.dataset.Label)

    # train a baseline model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = CNN_CIFAR10()
    net.to(device)
    epochs = 70
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    summary = {"train loss": [],
               "test loss": [],
               "train acc": [],
               "test acc": []}
    best_test_acc = 0
    save_path = experiment_path + "/CNN_CIFAR10_baseline.pth"

    for epoch in range(1, epochs+1):
        net.train()
        for X_batch, Y_batch in trainloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # forward
            logits = net(X_batch)
            loss = loss_function(logits, Y_batch)
            # backward
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
            torch.save(net.state_dict(), save_path)
            print("Saved")

        # save model every 5 epochs
        if epoch % 5 == 0:
            torch.save(net.state_dict(), experiment_path + "/CNN_CIFAR10_baseline_epoch{}.pth".format(epoch))

    return summary

def TracInCP():
    trainloader, testloader = prepare_CIFAR10(mode="tt")
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label
    n_train = len(Y_train)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = CNN_CIFAR10()
    net.to(device)

    self_influence = np.zeros(n_train)
    for i in range(5, 51, 5):
        load_path = "experiment3/CNN_CIFAR10_baseline_epoch{}.pth".format(i)
        net.load_state_dict(torch.load(load_path, map_location=device))
        tracin_this = tracin_self(net,X_train, Y_train)
        self_influence += tracin_this
        print("Epoch {} finished".format(i))

    np.save("experiment3/self_influence.npy", self_influence)


def experiment(drop_ratio=0.1, mode="middle"):
    """
    :param drop_ratio: the ratio of dropped samples
    :param mode: which part to keep
    """
    trainloader, testloader = prepare_CIFAR10(mode="tt")
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label

    self_influence = np.load("experiment3/self_influence.npy", allow_pickle=True)
    n_train = len(self_influence)

    if mode == "small":
        n_keep = int(n_train * (1 - drop_ratio))
        keep_index = np.argsort(self_influence)[:n_keep]
        random_keep_index = list(np.random.permutation(n_train)[:n_keep])
    elif mode == "large":
        n_drop = int(n_train * drop_ratio)
        keep_index = np.argsort(self_influence)[n_drop:]
        random_keep_index = list(np.random.permutation(n_train)[n_drop:])
    elif mode == "middle":
        n_drop_half = int(n_train * drop_ratio / 2)
        keep_index = np.argsort(self_influence)[n_drop_half:-n_drop_half]
        random_keep_index = list(np.random.permutation(n_train)[n_drop_half:-n_drop_half])

    X_train_new = X_train[keep_index]
    Y_train_new = Y_train[keep_index]
    X_train_random = X_train[random_keep_index]
    Y_train_random = Y_train[random_keep_index]

    trainloader_new = DataLoader(mydataset(X_train_new, Y_train_new), batch_size=128, shuffle=True)
    trainloader_random = DataLoader(mydataset(X_train_random, Y_train_random), batch_size=128, shuffle=True)


    epochs = 70
    print("TracIn: {} samples kept".format(len(Y_train_new)))
    train_CNN_CIFAR10(epochs, trainloader_new, testloader, save_path = "experiment3/pick {}/CNN_CIFAR10_drop{}.pth".format(mode, int(drop_ratio * 100)))
    print("Random: {} samples kept".format(len(Y_train_random)))
    train_CNN_CIFAR10(epochs, trainloader_random, testloader, save_path = "experiment3/pick {}/CNN_CIFAR10_drop{}random.pth".format(mode, int(drop_ratio * 100)))


def check():
    trainloader, testloader = prepare_CIFAR10(mode="tt")
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label

    self_influence = np.load("experiment3/self_influence.npy", allow_pickle=True)
    rank = np.argsort(self_influence)
    for i in range(10):
        rank_mean = np.mean(rank[Y_train == i])
        print("Class: {}. Mean Rank: {}".format(i, rank_mean))


def evaluation(mode="middle"):
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
        net.load_state_dict(
            torch.load("experiment3/pick {}/CNN_CIFAR10_drop{}.pth".format(mode, int(i * 10)), map_location=device))
        _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
        test_acc_tracin.append(acc)
        # random
        net.load_state_dict(
            torch.load("experiment3/pick {}/CNN_CIFAR10_drop{}random.pth".format(mode, int(i * 10)), map_location=device))
        _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
        test_acc_random.append(acc)

    xrange = [0] + [0.1 * i for i in range(2, 10)]
    plt.plot(xrange, test_acc_tracin, '-bo', label="Pick {}".format(mode))
    plt.plot(xrange, test_acc_random, '-ro', label="Pick Random")
    plt.title("Different Pick Rule")
    plt.xlabel("drop ratio")
    plt.ylabel("test acc")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    experiment_path = "./experiment3/pick middle"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    # drop_ratios = [0.1*i for i in range(2, 10)]
    # for drop_ratio in drop_ratios:
    #     experiment(drop_ratio, mode="small")
    #     experiment(drop_ratio, mode="middle")

    evaluation("small")
