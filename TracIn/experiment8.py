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

from utils import mydataset, device, get_loss_acc
from model import CNN_CIFAR10
from train import train_CNN_CIFAR10
from influence_functions import tracin_self


def prepare_CIFAR10_wrong(mode="tt", seed=42, wrong=True):
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

    if wrong:
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


def save_TracIn_self(ckpts=[45, 27, 20]):
    trainloader, testloader = prepare_CIFAR10_wrong()
    scores = np.zeros(len(trainloader.dataset.Label))
    for epoch in ckpts:
        net = CNN_CIFAR10().to(device)
        net.load_state_dict(torch.load(f"model_weights/weights_wrong_70epochs/CNN_CIFAR10_epoch{epoch}.pth", map_location=device))
        X_train = trainloader.dataset.Data
        Y_train = trainloader.dataset.Label
        score = tracin_self(net, X_train, Y_train)
        scores += score
        print("Epoch {} finished".format(epoch))

    np.save(f"experiment8/TracIn_self_{ckpts[0]}_{ckpts[1]}_{ckpts[2]}.npy", scores)


def check_labels(ckpts=[45, 27, 20]):
    scores = np.load(f"experiment8/TracIn_self_{ckpts[0]}_{ckpts[1]}_{ckpts[2]}.npy")
    trainloader, testloader = prepare_CIFAR10_wrong(wrong=False)
    trainloader_wrong, testloader_wrong = prepare_CIFAR10_wrong()
    Y_train_true = trainloader.dataset.Label.numpy()
    Y_train_wrong = trainloader_wrong.dataset.Label.numpy()
    sort_index = np.argsort(scores)[::-1]
    check_ratios = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    checked = []
    for ratio in check_ratios:
        n_check = int(len(Y_train_true) * ratio)
        check_index = sort_index[:n_check]
        n_checked = np.sum(Y_train_wrong[check_index] != Y_train_true[check_index])
        checked.append(n_checked / 5000)

    plt.plot(check_ratios, checked, ".-", label=f"{ckpts[0]} {ckpts[1]} {ckpts[2]}")
    # plt.title(f"TracIn computed by ckpt {ckpts[0]} {ckpts[1]} {ckpts[2]}")
    # plt.show()


def save_loss_reduction_ratio():
    trainloader, testloader = prepare_CIFAR10_wrong()
    loss_epoch = []
    net = CNN_CIFAR10().to(device)
    for i in range(1, 71):
        net.load_state_dict(torch.load(f"model_weights/weights_wrong_70epochs/CNN_CIFAR10_epoch{i}.pth",
                                       map_location=device))
        loss, acc = get_loss_acc(net, trainloader)
        loss_epoch.append(loss)
    loss_epoch = np.array(loss_epoch)
    np.save("model_weights/weights_wrong_70epochs/loss_by_epoch.npy", loss_epoch)
    loss_reduction_ratio = (-np.diff(loss_epoch) / loss_epoch[:-1])
    np.save("model_weights/weights_wrong_70epochs/loss_reduction_ratio_by_epoch.npy", loss_reduction_ratio)
    print(np.argsort(loss_reduction_ratio))






if __name__ == "__main__":
    check_labels([1,2,3])
    check_labels([4,7,11])
    check_labels([5, 15, 25])
    check_labels([20, 40, 60])
    plt.legend()
    plt.show()


