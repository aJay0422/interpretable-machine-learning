"""
Select validation samples using entropy
"""
import os.path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


from utils import device, mydataset, get_loss_acc
from data import prepare_CIFAR10
from model import CNN_CIFAR10
from train import train_CNN_CIFAR10
from influence_functions import select_train


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
    sort_index = np.sort(entropies)

    if mode == "hard":
        return_index = sort_index[:n]
    elif mode == "easy":
        return_index = sort_index[-n:]
    return {"X": X_val[return_index], "Y": Y_val[return_index]}


def experiment():
    experiment_path = "./experiment5"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    trainloader, valloader, testloader = prepare_CIFAR10()
    net = CNN_CIFAR10()
    methodV = ["hard", "easy"]
    methodT = ["T1", "T2", "T5", "T6"]
    for mv in methodV:
        for mt in methodT:
            # select validation and train
            selected_val = select_validation(net, valloader, mode=mv)
            selected_train = select_train(net, trainloader, selected_val, method=mt)
            print("train selected")

            # create a new trainloader
            selected_trainset = mydataset(selected_train["X"], selected_train["Y"])
            selected_trainloader = DataLoader(selected_trainset, batch_size=256, shuffle=True)

            # load trained model from 3rd epoch
            net.load_state_dict(torch.load("model_weights/CNN_CIFAR10_epoch3.pth", map_location=device))

            # train model and save checkpoint
            n_train = len(selected_trainset.Label)
            print(f"{mv} {mt}", end="  ")
            print("Trained on a dataset of {} samples".format(n_train))
            save_path = experiment_path + "/CNN_CIFAR10_{}{}.pth".format(mv, mt)
            train_CNN_CIFAR10(70, selected_trainloader, valloader, save_path=save_path)


def evaluation():
    MV = ["hard", "easy"]
    MT = ["T1", "T2", "T5", "T6"]
    trainloader, valloader, testloader = prepare_CIFAR10()
    net = CNN_CIFAR10().to(device)
    for mv in MV:
        for mt in MT:
            weight_path = f"experiment5/CNN_CIFAR10_{mv}{mt}.pth"
            net.load_state_dict(torch.load(weight_path, map_location=device))
            _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
            print(f"{mv} {mt} {acc}")


def random_keep(keep_nums):
    for keep_num in keep_nums:
        trainloader, valloader, testloader = prepare_CIFAR10()
        n_train = len(trainloader.dataset.Label)
        keep_index = list(np.random.permutation(n_train)[:keep_num])
        X_train_keep = trainloader.dataset.Data[keep_index]
        Y_train_keep = trainloader.dataset.Label[keep_index]
        trainloader = DataLoader(mydataset(X_train_keep, Y_train_keep), batch_size=128, shuffle=True)
        save_path = "experiment5/CNN_CIFAR10_rndkeep{}.pth".format(keep_num)
        train_CNN_CIFAR10(70, trainloader, valloader, save_path=save_path)


def evaluation2():
    keep_nums = [8946, 10000, 11017, 28983, 31054]
    trainloader, valloader, testloader = prepare_CIFAR10()
    net = CNN_CIFAR10().to(device)
    for keep_num in keep_nums:
        weight_path = "experiment5/CNN_CIFAR10_rndkeep{}.pth".format(keep_num)
        net.load_state_dict(torch.load(weight_path, map_location=device))
        _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
        print(keep_num, acc)

if __name__ == "__main__":
    evaluation2()

    stop = None