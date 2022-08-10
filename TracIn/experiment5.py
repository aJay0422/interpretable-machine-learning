"""
Select validation samples using entropy
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import device, mydataset, get_loss_acc
from data import prepare_CIFAR10
from model import CNN_CIFAR10
from train import train_CNN_CIFAR10
from influence_functions import tracin_multi_multi, select_train


def select_validation(net, valloader, n=30, mode="hard"):
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10_epoch3.pth", map_location=device))
    net.to(device)

    # calculate entropy loss for each sample in valloader
    loss_function = nn.CrossEntropyLoss()
    entropies = []
    X_val = valloader.dataset.Data
    Y_val = valloader.dataset.Label
    with torch.no_grad():
        net.eval()
        for X, Y in zip(X_val, Y_val):
            X = X.to(device).unsqueeze(0)
            Y = Y.to(device).unsqueeze(0)
            logit = net(X)
            loss = loss_function(logit, Y)
            entropies.append(loss.cpu().item())
    entropies = np.array(entropies)
    sort_index = np.argsort(entropies)

    if mode == "hard":   # choose samples with high loss
        return_index = []
        for i in range(10):
            return_index.append(sort_index[Y_val[sort_index] == i][-(n // 10):])
        return_index = np.hstack(return_index)
    elif mode == "easy":   # choose samples with low loss
        return_index = []
        for i in range(10):
            return_index.append(sort_index[Y_val[sort_index] == i][:(n // 10)])
        return_index = np.hstack(return_index)

    # softmax
    cls_weight = []
    for i in range(10):
        weight = np.mean(entropies[Y_val == i])
        cls_weight.append(weight)
    cls_weight = np.array(cls_weight) / np.sqrt(10)
    cls_weight = np.exp(cls_weight)
    cls_weight = cls_weight / np.sum(cls_weight)

    return {"X": X_val[return_index], "Y": Y_val[return_index]}, cls_weight


def experiment():
    experiment_path = "./experiment5"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    trainloader, valloader, testloader = prepare_CIFAR10()
    net = CNN_CIFAR10()
    methodV = ["hard", "easy"]
    methodT = ["T11", "T12"]
    for mv in methodV:
        for mt in methodT:
            setting_path = experiment_path + "/{}{}".format(mv, mt)
            if not os.path.exists(setting_path):
                os.mkdir(setting_path)

            # select validation and train
            selected_val = select_validation(net, valloader, mode=mv)
            selected_train = select_train(net, trainloader, selected_val, method=mt)
            np.savez(setting_path + "/selected_val.npz", X=selected_val["X"], Y=selected_val["Y"])
            np.savez(setting_path + "/selected_train.npz", X=selected_train["X"], Y=selected_train["Y"])
            print("train selected")

            # create a new trainloader
            selected_trainset = mydataset(selected_train["X"], selected_train["Y"])
            selected_trainloader = DataLoader(selected_trainset, batch_size=256, shuffle=True)

            # train model and save checkpoint
            n_train = len(selected_trainset.Label)
            print(f"{mv} {mt}", end="  ")
            print("Trained on a dataset of {} samples".format(n_train))
            save_path = setting_path + "/CNN_CIFAR10_{}{}.pth".format(mv, mt)
            train_CNN_CIFAR10(70, selected_trainloader, valloader, load_path="model_weights/CNN_CIFAR10_epoch3.pth", save_path=save_path)


def evaluation():
    MV = ["hard", "easy"]
    MT = ["T11", "T12"]
    trainloader, valloader, testloader = prepare_CIFAR10()
    net = CNN_CIFAR10().to(device)
    for mv in MV:
        for mt in MT:
            weight_path = f"experiment5/{mv}{mt}/CNN_CIFAR10_{mv}{mt}.pth"
            net.load_state_dict(torch.load(weight_path, map_location=device))
            _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
            file = np.load(f"experiment5/{mv}{mt}/selected_train.npz", allow_pickle=True)
            n_ST = len(file["Y"])
            print(f"{mv} {mt} {acc} {n_ST}")


def random_keep(keep_nums):
    for keep_num in keep_nums:
        for i in range(5):
            trainloader, valloader, testloader = prepare_CIFAR10()
            n_train = len(trainloader.dataset.Label)
            keep_index = list(np.random.permutation(n_train)[:keep_num])
            X_train_keep = trainloader.dataset.Data[keep_index]
            Y_train_keep = trainloader.dataset.Label[keep_index]
            trainloader = DataLoader(mydataset(X_train_keep, Y_train_keep), batch_size=128, shuffle=True)
            save_path = "experiment5/CNN_CIFAR10_rndkeep{}_{}.pth".format(keep_num, i+1)
            train_CNN_CIFAR10(70, trainloader, valloader, load_path="model_weights/CNN_CIFAR10_epoch3.pth", save_path=save_path)


def evaluation2(keep_nums):
    trainloader, valloader, testloader = prepare_CIFAR10()
    net = CNN_CIFAR10().to(device)
    for keep_num in keep_nums:
        accs = []
        for i in range(1, 6):
            weight_path = "experiment5/CNN_CIFAR10_rndkeep{}_{}.pth".format(keep_num, i)
            net.load_state_dict(torch.load(weight_path, map_location=device))
            _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
            accs.append(acc)
        m = np.mean(accs)
        v = np.std( accs)
        print(keep_num, m, v)


def val_loss_analysis(net, valloader):
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10.pth", map_location=device))
    net.to(device)

    # calculate entropy loss for each sample in valloader
    loss_function = nn.CrossEntropyLoss()
    entropies = []
    X_val = valloader.dataset.Data
    Y_val = valloader.dataset.Label
    with torch.no_grad():
        net.eval()
        for X, Y in zip(X_val, Y_val):
            X = X.to(device).unsqueeze(0)
            Y = Y.to(device).unsqueeze(0)
            logit = net(X)
            loss = loss_function(logit, Y)
            entropies.append(loss.cpu().item())
    entropies = np.array(entropies)
    df = pd.DataFrame({"entropy": entropies, "label": Y_val})
    df.boxplot(column="entropy", by="label", showfliers=False, grid=False)
    plt.xticks(np.arange(1, 11), ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"], rotation=45)
    plt.show()


def val_loss_decrease_visualize(valloader):
    net = CNN_CIFAR10().to(device)
    loss_function = nn.CrossEntropyLoss()
    X_val = valloader.dataset.Data
    Y_val = valloader.dataset.Label
    cls_entropy_by_time = []
    for epoch in range(55, 71):
        net.load_state_dict(torch.load(f"model_weights/CNN_CIFAR10_epoch{epoch}.pth", map_location=device))
        entropies = []
        with torch.no_grad():
            net.eval()
            for X, Y in zip(X_val, Y_val):
                X = X.to(device).unsqueeze(0)
                Y = Y.to(device).unsqueeze(0)
                logit = net(X)
                loss = loss_function(logit, Y)
                entropies.append(loss.cpu().item())
        entropies = np.array(entropies)
        cls_entropy = []
        for i in range(10):
            m = np.mean(entropies[Y_val == i])
            cls_entropy.append(m)
        cls_entropy_by_time.append(cls_entropy)

    cls_entropy_by_time = np.vstack(cls_entropy_by_time)
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    for i in range(10):
        plt.plot(cls_entropy_by_time[:, i], label=labels[i])
    plt.legend(prop={"size": 6})
    plt.show()



if __name__ == "__main__":
    net = CNN_CIFAR10()
    trainloader, valloader, testloader = prepare_CIFAR10()
    SV, cls_weight = select_validation(net, valloader)
    print(cls_weight)

    stop = None