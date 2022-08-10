"""
Select validation samples using entropy. Then select train samples according to class importance
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from utils import device, mydataset
from influence_functions import tracin_multi_multi
from data import prepare_CIFAR10
from model import CNN_CIFAR10
from train import train_CNN_CIFAR10


def select_validation(net, valloader, n=30, mode="hard", d=10):
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

    if mode == "hard":
        return_index = []
        for i in range(10):
            return_index.append(sort_index[Y_val[sort_index] == i][-(n // 10): ])
        return_index = np.hstack(return_index)
    elif mode == "easy":
        return_index = []
        for i in range(10):
            return_index.append(sort_index[Y_val[sort_index] == i][: (n // 10)])
        return_index = np.hstack(return_index)

    # softmax
    cls_weight = []
    for i in range(10):
        weight = np.mean(entropies[Y_val == i])
        cls_weight.append(weight)
    cls_weight = np.array(cls_weight) / np.sqrt(d)
    cls_weight = np.exp(cls_weight)
    cls_weight = cls_weight / np.sum(cls_weight)

    return {"X": X_val[return_index], "Y": Y_val[return_index]}, cls_weight


def selecet_train(net, trainloader, selected_val, cls_weight, mode="high"):
    """
    :param net: the neural net for TracIn calculation
    :param trainloader: all train samples
    :param selected_val: selected validation samples
    :param cls_weight: importance of each class
    :return: selected train samples
    """
    # prepare model
    net.to(device)

    # prepare data
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label
    n_train = len(Y_train)

    X_val = selected_val["X"]
    Y_val = selected_val["Y"]

    # compute TracIn
    TracIn_by_cls = {}
    for cls in range(10):
        cls_idx_train = (Y_train == cls)
        cls_idx_val = (Y_val == cls)
        n_train_tmp = len(Y_train[cls_idx_train])
        n_val_tmp = len(Y_val[cls_idx_val])
        TracIn_by_cls[cls] = np.zeros((n_train_tmp, n_val_tmp))
        for i in range(1, 4):
            # load weight
            net.load_state_dict(torch.load(f"model_weights/CNN_CIFAR10_epoch{i}.pth", map_location=device))
            TracIn_by_cls[cls] += tracin_multi_multi(net, X_train[cls_idx_train], X_val[cls_idx_val],
                                                     Y_train[cls_idx_train], Y_val[cls_idx_val])

    cls_weight = cls_weight / np.max(cls_weight)
    n_keep_cls = [int(4000 * weight) for weight in cls_weight]

    X_selected = []
    Y_selected = []
    if mode == "high":
        for cls in range(10):
            TracIn_mean = np.mean(TracIn_by_cls[cls], axis=1)
            index = list(np.argsort(TracIn_mean)[-n_keep_cls[cls]: ])
            X_selected.append(X_train[Y_train == cls][index])
            Y_selected.append(Y_train[Y_train == cls][index])
    elif mode == "low":
        for cls in range(10):
            TracIn_mean = np.mean(TracIn_by_cls[cls], axis=1)
            index = list(np.argsort(TracIn_mean)[: n_keep_cls[cls]])
            X_selected.append(X_train[Y_train == cls][index])
            Y_selected.append(Y_train[Y_train == cls][index])

    X_selected = np.concatenate(X_selected, axis=0)
    Y_selected = np.concatenate(Y_selected, axis=0)
    return {"X": X_selected, "Y": Y_selected}


def experiment():
    experiment_path = "./experiment6"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    MV = ["hard", "easy"]
    MT = ["high", "low"]
    trainloader, valloader, testloader = prepare_CIFAR10()
    net = CNN_CIFAR10().to(device)
    for mv in MV:
        for mt in MT:
            setting_path = experiment_path + "/{}{}".format(mv, mt)
            if not os.path.exists(setting_path):
                os.mkdir(setting_path)

            # select validation and train
            selected_val, cls_weight = select_validation(net, valloader, mode=mv)
            selected_train = selecet_train(net, trainloader, selected_val, cls_weight, mode=mt)
            np.savez(setting_path + "/selected_val.npz", X=selected_val["X"], Y=selected_val["Y"])
            np.savez(setting_path + "/selected_train.npz", X=selected_train["X"], Y=selected_train["Y"])
            print("train selected")

            # create a new trainloader
            selected_trainset = mydataset(selected_train["X"], selected_train["Y"])
            selected_trainloader = DataLoader(selected_trainset, batch_size=256, shuffle=True)

            # train model and save checkpoint
            n_train = len(selected_trainset.Label)
            print(f"{mv}{mt}", end=" ")
            print("Trained on a dataset of {} samples".format(n_train))
            save_path = setting_path + "/CNN_CIFAR10_{}{}.pth".format(mv, mt)
            train_CNN_CIFAR10(70, selected_trainloader, valloader,
                              load_path="model_weights/CNN_CIFAR10_epoch3.pth", save_path=save_path)



if __name__ == "__main__":
    experiment()