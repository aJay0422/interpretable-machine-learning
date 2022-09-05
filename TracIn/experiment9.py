"""
Calculate TracIn during the training process
"""
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import time

from model import CNN_CIFAR10
from data import prepare_CIFAR10
from train import train_CNN_CIFAR10
from utils import get_loss_acc, device, get_tracin, mydataset


class mydataset_w_index(Dataset):
    def __init__(self, X, Y):
        self.Data = torch.Tensor(X)
        self.Label = torch.LongTensor(Y)
        self.index = torch.LongTensor(np.arange(len(Y)))

    def __getitem__(self, item):
        return self.Data[item], self.Label[item], self.index[item]

    def __len__(self):
        return len(self.Label)


def prepare_CIFAR10_w_index(mode="tvt"):
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # load CIFAR10 train and test dataset
    train_set = torchvision.datasets.CIFAR10(root="datasets/CIFAR10/", train=True,
                                             download=False, transform=data_transform["train"])
    test_set = torchvision.datasets.CIFAR10(root="datasets/CIFAR10/", train=False,
                                            download=False, transform=data_transform["val"])
    X_train = train_set.data.transpose((0, 3, 1, 2))
    X_test = test_set.data.transpose((0, 3, 1, 2))
    Y_train = train_set.targets
    Y_test = test_set.targets

    if mode == "tt":
        train_set = mydataset_w_index(X_train, Y_train)
        test_set = mydataset_w_index(X_test, Y_test)
        trainloader = DataLoader(train_set, batch_size=64,
                                 shuffle=True)
        testloader = DataLoader(test_set, batch_size=64,
                                shuffle=False)
        print(len(X_train), len(X_test))
        return trainloader, testloader

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, stratify=Y_train, test_size=0.2,
                                                      random_state=42)  # split 10000 validation data

    train_set = mydataset_w_index(X_train, Y_train)
    val_set = mydataset_w_index(X_val, Y_val)
    test_set = mydataset_w_index(X_test, Y_test)

    trainloader = DataLoader(train_set, batch_size=64,
                             shuffle=True)
    valloader = DataLoader(val_set, batch_size=64,
                           shuffle=False)
    testloader = DataLoader(test_set, batch_size=64,
                            shuffle=False)
    print(len(X_train), len(X_val), len(X_test))
    return trainloader, valloader, testloader


def sanity_check():
    trainloader, testloader, valloader = prepare_CIFAR10_w_index()
    net = CNN_CIFAR10().to(device)
    for param in net.parameters():
        print(param[0])
        break
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    X_one = trainloader.dataset.Data[0].unsqueeze(0)
    Y_one = trainloader.dataset.Label[0].unsqueeze(0)

    logit = net(X_one.to(device))
    loss = loss_function(logit, Y_one.to(device))
    optimizer.zero_grad()
    loss.backward()
    grad_train = []
    for param in net.parameters():
        grad_train.append(param.grad)

    for param in net.parameters():
        print(param[0])
        break
    logit = net(X_one.to(device))
    loss = loss_function(logit, Y_one.to(device))
    grad_train2 = grad(loss, net.parameters())
    grad_train2 = [g for g in grad_train2]

    stop = None



def train_CNN_CIFAR10_w_index(epochs,
                              trainloader,
                              testloader, selected_val,
                              seed=20220718, get_summary=False,
                              load_path=None, save_path="experiment9/CNN_CIFAR10.pth",
                              TracIn_save_path="experiment9/TracIn_all.npy"):
    # setup
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if get_summary:
        summary = {"train loss": [],
                   "test loss": [],
                   "train acc": [],
                   "test acc": []}

    # prepare model
    net = CNN_CIFAR10(num_classes=10)
    net.to(device)

    if load_path is not None:
        net.load_state_dict(torch.load(load_path, map_location=device))

    # train model
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    print("Trained on {}".format(device))
    net.to(device)

    # initial
    with torch.no_grad():
        net.eval()
        # evaluate train
        train_loss, train_acc = get_loss_acc(net, trainloader)
        # evaluate test
        test_loss, test_acc = get_loss_acc(net, testloader)

    print("Epoch{}/{}  train loss: {}  test loss: {}  train acc: {}  test acc: {}".format(0, epochs,
                                                                                          train_loss, test_loss,
                                                                                          train_acc, test_acc))

    best_test_acc = 0
    X_val = selected_val["X"]
    Y_val = selected_val["Y"]
    n_train = len(trainloader.dataset.Label)
    n_sv = len(Y_val)
    TracIn_all = np.zeros((n_train, n_sv))

    for epoch in range(epochs):
        net.to(device)
        time_summary = {"forward":0,
                        "backward":0,
                        "update":0,
                        "extract grad train":0,
                        "compute grad test":0,
                        "compute TracIn":0,
                        "total":0}
        ptime_total = time.time()
        for X_batch, Y_batch, index_batch in trainloader:
            net.train()
            batch_size = len(index_batch)
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            ptime = time.time()
            logits = net(X_batch)
            time_summary["forward"] += (time.time() - ptime)
            loss = loss_function(logits, Y_batch)
            # backward
            optimizer.zero_grad()
            ptime = time.time()
            loss.backward()
            time_summary["backward"] += (time.time() - ptime)
            ptime = time.time()
            optimizer.step()
            time_summary["update"] += (time.time() - ptime)

            ptime = time.time()
            optimizer.zero_grad()
            # extract gradient for this train batch
            grad_train = []
            for param in net.parameters():
                grad_train.append(param.grad)
            time_summary["extract grad train"] += (time.time() - ptime)

            # calculate gradient for each sample in selected validation
            net.eval()
            scores = []
            for i in range(n_sv):
                ptime = time.time()
                z_test = torch.Tensor(X_val[i]).unsqueeze(dim=0)
                label_test = torch.LongTensor([Y_val[i]])
                logits_test = net(z_test.to(device))
                loss_test = loss_function(logits_test, label_test.to(device))
                grad_test = grad(loss_test, net.parameters())
                grad_test = [g for g in grad_test]
                time_summary["compute grad test"] += (time.time() - ptime)

                ptime = time.time()
                score = get_tracin(grad_train, grad_test)
                scores.append(score.cpu() / batch_size)
                time_summary["compute TracIn"] += (time.time() - ptime)
            scores = np.array(scores)
            TracIn_all[index_batch, :] += scores
        time_summary["total"] += (time.time() - ptime_total)
        print("Total time for 1 epoch is {:.2f}s".format(time_summary["total"]))
        for name in time_summary:
            if name == "total":
                continue
            print("{} uses {:.2f}% of total time".format(name, time_summary[name] / time_summary["total"] * 100))




        # # evaluate
        # with torch.no_grad():
        #     net.eval()
        #     # evaluate train
        #     train_loss, train_acc = get_loss_acc(net, trainloader)
        #     # evaluate test
        #     test_loss, test_acc = get_loss_acc(net, testloader)
        #
        # print("Epoch{}/{}  train loss: {}  test loss: {}  train acc: {}  test acc: {}".format(epoch + 1, epochs,
        #                                                                                       train_loss, test_loss,
        #                                                                                       train_acc, test_acc))

    #     if get_summary:
    #         summary["train loss"].append(train_loss)
    #         summary["test loss"].append(test_loss)
    #         summary["train acc"].append(train_acc)
    #         summary["test acc"].append(test_acc)
    #
    #     # save model weights of it's the best
    #     if test_acc >= best_test_acc:
    #         best_test_acc = test_acc
    #         torch.save(net.state_dict(), save_path)
    #         print("Saved")
    #
    # np.save(TracIn_save_path, TracIn_all)
    #
    # if get_summary:
    #     return summary


def experiment_TracIn_all(mode="high"):
    experiment_path = "experiment9"

    keep_ratios = [0.1, 0.2, 0.4, 0.6, 0.8]
    trainloader, valloader, testloader = prepare_CIFAR10()

    for keep_ratio in keep_ratios:
        TracIn_all = np.load("experiment9/TracIn_all.npy")
        X_train = trainloader.dataset.Data
        Y_train = trainloader.dataset.Label
        n_train = len(Y_train)
        n_keep = int(n_train * keep_ratio)

        if mode == "high":
            selected_index = list(np.argsort(np.mean(TracIn_all, axis=1))[::-1][:n_keep])
        elif mode == "mid":
            drop_half = int((n_train - n_keep) // 2)
            selected_index = list(np.argsort(np.mean(TracIn_all, axis=1))[::-1][drop_half:-drop_half])

        X_selected = X_train[selected_index]
        Y_selected = Y_train[selected_index]

        # create a new trainloader
        selected_trainloader = DataLoader(mydataset(X_selected, Y_selected),
                                          batch_size=256)

        save_path = experiment_path + f"/CNN_CIFAR10_TracIn_all_{mode}_{keep_ratio}.pth"
        train_CNN_CIFAR10(100, selected_trainloader, valloader,
                          save_path=save_path)


def evaluation_TracIn_all():
    keep_ratios = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
    accs_all = {}
    trainloader, valloader, testloader = prepare_CIFAR10()

    # get baseline acc
    net = CNN_CIFAR10().to(device)
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10.pth", map_location=device))
    _, baseline_acc = get_loss_acc(net, testloader)

    # get random keep accuracy
    accs = np.zeros(len(keep_ratios))
    accs[-1] = baseline_acc
    for i, keep_ratio in enumerate(keep_ratios[:-1]):
        net.load_state_dict(
            torch.load(f"experiment7/random_keep/CNN_CIFAR10_rndkeep{keep_ratio}_1.path", map_location=device)
        )
        _, acc = get_loss_acc(net, testloader)
        accs[i] = acc
    accs_all["random"] = accs

    # get TracIn live accuarcy
    name = "TracIn live high"
    accs = np.zeros(len(keep_ratios))
    accs[-1] = baseline_acc
    for i, keep_ratio in enumerate(keep_ratios[:-1]):
        net.load_state_dict(torch.load(f"experiment9/CNN_CIFAR10_TracIn_all_high_{keep_ratio}.pth", map_location=device))
        _, acc = get_loss_acc(net, testloader)
        accs[i] = acc
    accs_all[name] = accs

    name = "TracIn live mid"
    accs = np.zeros(len(keep_ratios))
    accs[-1] = baseline_acc
    for i, keep_ratio in enumerate(keep_ratios[:-1]):
        net.load_state_dict(
            torch.load(f"experiment9/CNN_CIFAR10_TracIn_all_mid_{keep_ratio}.pth", map_location=device))
        _, acc = get_loss_acc(net, testloader)
        accs[i] = acc
    accs_all[name] = accs

    # get TracIn ckpt accuracy
    name = "TracIn ckpt"
    accs = np.zeros(len(keep_ratios))
    accs[-1] = baseline_acc
    for i, keep_ratio in enumerate(keep_ratios[:-1]):
        net.load_state_dict(torch.load(f"experiment7/TracIn_all/CNN_CIFAR10_TracIN_all_{keep_ratio}.pth",
                                       map_location=device))
        _, acc = get_loss_acc(net, testloader)
        accs[i] = acc
    accs_all[name] = accs

    # get TracIn ckpt by class accuracy
    name = "TracIn ckpt by cls"
    accs = np.zeros(len(keep_ratios))
    accs[-1] = baseline_acc
    for i, keep_ratio, in enumerate(keep_ratios[:-1]):
        net.load_state_dict(torch.load(f"experiment7/from_beginning(loss)/randomhigh{keep_ratio}/CNN_CIFAR10.pth",
                                       map_location=device))
        _, acc = get_loss_acc(net, testloader)
        accs[i] = acc
    accs_all[name] = accs

    fig = plt.figure(figsize=(10, 10))
    for name in accs_all:
        plt.plot(keep_ratios, accs_all[name], ".-", label=name)
        print(name)
        print(accs_all[name])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    evaluation_TracIn_all()


