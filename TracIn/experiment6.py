"""
Select validation samples using entropy. Then select train samples according to class importance
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from utils import device, mydataset, get_loss_acc
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


def experiment(d=9, from_beginning=False):
    if from_beginning:
        experiment_path = "./experiment6/from_beginning"
    else:
        experiment_path = "./experiment6"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    MV = ["hard", "easy"]
    MT = ["high", "low"]
    trainloader, valloader, testloader = prepare_CIFAR10()
    net = CNN_CIFAR10().to(device)
    for mv in MV:
        for mt in MT:
            setting_path = experiment_path + "/{}{}{}".format(mv, mt, d)
            if not os.path.exists(setting_path):
                os.mkdir(setting_path)

            # select validation and train
            selected_val, cls_weight = select_validation(net, valloader, mode=mv, d=d)
            selected_train = selecet_train(net, trainloader, selected_val, cls_weight, mode=mt)
            np.savez(setting_path + "/selected_val.npz", X=selected_val["X"], Y=selected_val["Y"])
            np.savez(setting_path + "/selected_train.npz", X=selected_train["X"], Y=selected_train["Y"])
            print("train selected")

            # create a new trainloader
            selected_trainset = mydataset(selected_train["X"], selected_train["Y"])
            selected_trainloader = DataLoader(selected_trainset, batch_size=256, shuffle=True)

            # train model and save checkpoint
            n_train = len(selected_trainset.Label)
            print(f"{mv}{mt}{d}", end=" ")
            print("Trained on a dataset of {} samples".format(n_train))
            save_path = setting_path + "/CNN_CIFAR10_{}{}{}.pth".format(mv, mt, d)
            if from_beginning:
                load_path = None
            else:
                load_path = "model_weights/CNN_CIFAR10_epoch3.pth"
            train_CNN_CIFAR10(70, selected_trainloader, valloader,
                              load_path=load_path, save_path=save_path)


def n_SV():
    net = CNN_CIFAR10().to(device)
    net.load_state_dict(torch.load(f"model_weights/CNN_CIFAR10_epoch3.pth", map_location=device))
    trainloader, valloader, testloader = prepare_CIFAR10()

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

    # softmax
    cls_weight = []
    for i in range(10):
        weight = np.mean(entropies[Y_val == i])
        cls_weight.append(weight)

    ds = [0.03, 0.1, 0.3, 1, 3, 9]
    for d in ds:
        cls_ratio = np.array(cls_weight) / np.sqrt(d)
        cls_ratio = np.exp(cls_ratio)
        cls_ratio = cls_ratio / np.sum(cls_ratio)
        cls_ratio = cls_ratio / np.max(cls_ratio)
        n_keep_cls = [int(4000 * ratio) for ratio in cls_ratio]
        n_sv = np.sum(n_keep_cls)
        print(f"d = {d}  n_SV = {n_sv}")


def evaluation(from_beginning=False):
    mv = ["hard", "easy"]
    mt = ["high", "low"]
    ds = [0.03, 0.1, 0.3, 1, 3, 9]
    net = CNN_CIFAR10().to(device)
    trainloader, valloader, testloader = prepare_CIFAR10()
    for v in mv:
        for t in mt:
            for d in ds:
                if from_beginning:
                    setting_path = f"experiment6/from_beginning/{v}{t}{d}"
                else:
                    setting_path = f"experiment6/{v}{t}{d}"

                weight_path = setting_path + f"/CNN_CIFAR10_{v}{t}{d}.pth"
                net.load_state_dict(torch.load(weight_path, map_location=device))
                _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
                file = np.load(setting_path + "/selected_train.npz", allow_pickle=True)
                n_sv = len(file["Y"])
                print("{} {}  Trained on {} samples  Test Acc: {:.2f}%".format(v, t, n_sv, acc * 100))

    # baseline
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10.pth", map_location=device))
    _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
    print("Baseline  Trained on {} samples  Test Acc: {:.2f}%".format(40000, acc * 100))


def random_keep(keep_nums=[34298, 30959, 26420, 20564, 15471, 11130], from_beginning=False):
    for keep_num in keep_nums:
        for i in range(5):
            trainloader, valloader, testloader = prepare_CIFAR10()
            n_train = len(trainloader.dataset.Label)
            keep_index = list(np.random.permutation(n_train)[:keep_num])
            X_train_keep = trainloader.dataset.Data[keep_index]
            Y_train_keep = trainloader.dataset.Label[keep_index]
            trainloader = DataLoader(mydataset(X_train_keep, Y_train_keep), batch_size=128, shuffle=True)

            if from_beginning:
                load_path = None
                save_path = "experiment6/CNN_CIFAR10_rndkeep{}_{}_frombegin.pth".format(keep_num, i + 1)
            else:
                load_path = "model_weights/CNN_CIFAR10_epoch3.pth"
                save_path = "experiment6/CNN_CIFAR10_rndkeep{}_{}.pth".format(keep_num, i + 1)
            train_CNN_CIFAR10(70, trainloader, valloader, load_path=load_path, save_path=save_path)


def evaluate_random_keep(keep_nums=[34298, 30959, 26420, 20564, 15471, 11130], from_beginning=False):
    trainloader, valloader, testloader = prepare_CIFAR10()
    net = CNN_CIFAR10().to(device)
    for keep_num in keep_nums:
        accs = []
        for i in range(1, 6):
            if from_beginning:
                weight_path = "experiment6/CNN_CIFAR10_rndkeep{}_{}_frombegin.pth".format(keep_num, i)
            else:
                weight_path = "experiment6/CNN_CIFAR10_rndkeep{}_{}.pth".format(keep_num, i)
            net.load_state_dict(torch.load(weight_path, map_location=device))
            _, acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())
            accs.append(acc)
        m = np.mean(accs)
        v = np.std(accs)
        print(keep_num, m, v)


def class_dist_check(d=9):
    experiment_path = "experiment6/"
    svs = ["hard", "easy"]
    sts = ["high", "low"]
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))

    for i, sv in enumerate(svs):
        for j, st in enumerate(sts):
            folder_name = sv + st + str(d)
            folder_path = experiment_path + folder_name
            file = np.load(folder_path + "/selected_train.npz", allow_pickle=True)
            Y = file["Y"]
            ax[i,j].hist(Y)
            ax[i,j].title.set_text("{} {}".format(sv, st))
    plt.title("n samples = {}".format(len(Y)))
    plt.show()


def class_acc_check(d=9):
    trainloader, valloader, testloader = prepare_CIFAR10()
    experiment_path = "experiment6/"
    svs = ["hard", "easy"]
    sts = ["high", "low"]
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))

    for i, sv in enumerate(svs):
        for j, st in enumerate(sts):
            folder_name = sv + st + str(d)
            folder_path = experiment_path + folder_name
            net = CNN_CIFAR10().to(device)
            net.load_state_dict(torch.load(folder_path + "/CNN_CIFAR10_" + folder_name + ".pth", map_location=device))
            predictions = []
            with torch.no_grad():
                net.eval()
                for X_batch, Y_batch in testloader:
                    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                    logits = net(X_batch)
                    y_pred = torch.argmax(logits, dim=1).cpu()
                    predictions.append(y_pred)

            predictions = torch.hstack(predictions)
            accs = []
            Y_test = testloader.dataset.Label
            for cls in range(10):
                idx = (Y_test == cls)
                acc = np.mean(predictions[idx].eq(Y_test[idx]).numpy())
                accs.append(acc)

            ax[i, j].plot(accs, "b.")
            ax[i, j].title.set_text("{} {}".format(sv, st))

    plt.show()





if __name__ == "__main__":
    for i in [9, 3, 1, 0.3, 0.1, 0.03]:
        class_dist_check(d=i)
        class_acc_check(d=i)
