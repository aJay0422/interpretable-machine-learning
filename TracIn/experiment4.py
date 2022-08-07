import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from utils import mydataset, get_loss_acc
from model import CNN_CIFAR10
from data import prepare_CIFAR10
from train import train_CNN_CIFAR10


def drop_validation(valloader, keep_ratio=0.1, mode="random"):
    X_val, Y_val = valloader.dataset.Data, valloader.dataset.Label
    n_val = len(Y_val)
    n_keep = int(n_val * keep_ratio)
    if mode == "random":
        # np.random.seed(42)
        keep_index = np.random.permutation(n_val)[:n_keep]

    valloader = DataLoader(mydataset(X_val[keep_index], Y_val[keep_index]), batch_size=64, shuffle=False)
    return valloader

def evaluation(load_path, testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = CNN_CIFAR10()
    net.to(device)
    net.load_state_dict(torch.load(load_path, map_location=device))
    _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
    return acc


def experiment():
    keep_ratios = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    for keep_ratio in keep_ratios:

        # prepare data
        trainloader, valloader, testloader = prepare_CIFAR10()

        # get net valloader
        mode = "random"
        n_keep = int(10000 * keep_ratio)
        valloader = drop_validation(valloader, keep_ratio=keep_ratio, mode=mode)

        # train model
        experiment_path = "./experiment4"
        if not os.path.exists(experiment_path):
            os.mkdir(experiment_path)

        epochs = 70
        train_CNN_CIFAR10(epochs, trainloader, valloader,
                          save_path=experiment_path + "/CNN_CIFAR10_{}_{}.pth".format(mode, n_keep))

    trainloader, valloader, testloader = prepare_CIFAR10()
    accs = []
    for keep_ratio in keep_ratios:
        file_path = "./experiment4/CNN_CIFAR10_random_{}.pth".format(int(keep_ratio * 10000))
        acc = evaluation(file_path, testloader)
        accs.append(acc)

    for keep_ratio, acc in zip(keep_ratios, accs):
        print("Keep Ratio: {}  Test Acc: {:.2f}%".format(keep_ratio, acc * 100))

    return accs


if __name__ == "__main__":
    all_trials = np.load("experiment4/all_trials.npy", allow_pickle=True)
    trials_mean = np.mean(all_trials, axis=0)
    trials_max = np.max(all_trials, axis=0)
    trials_min = np.min(all_trials, axis=0)
    fig, ax = plt.subplots()
    ax.fill_between(np.arange(7), trials_max, trials_min, alpha=0.2)
    ax.plot(np.arange(7), trials_mean, "--bo")
    ax.set_xticks(np.arange(7), [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1])
    ax.set_xlabel("Kept Proportion")
    ax.set_ylabel("Test Accuracy")
    ax.text(0.001, 0.83, "max", fontsize=15, color="red")
    ax.text(0.001, 0.68, "min", fontsize=15, color="red")
    ax.text(0.3, 0.81, "mean", fontsize=15, color="red")
    plt.show()



