import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import time
import os

from model import CNN_CIFAR10
from data import prepare_CIFAR10
from utils import get_tracin


def tracin_single_single(net, z_train, z_test, label_train, label_test, loss_function=nn.CrossEntropyLoss()):
    """
    Compute the tracin score for 2 samples
    :param net: the corresponding model
    :param z_train: 1st sample, a torch tensor of shape (3, H, W)
    :param z_test: 2nd sample, a torch tensor of shape (3, H, W)
    :param label_train: the label corresponding to the train sample
    :param label_test: the label corresponding to the test sample
    :param loss_function: loss function to calculate loss on a single sample
    :return: the tracin score
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prepare model
    net.to(device)

    # prepare data
    z_train = z_train.unsqueeze(dim=0)
    z_test = z_test.unsqueeze(dim=0)
    if label_train.size() != torch.Size([1]):
        label_train = label_train.unsqueeze(dim=0)
    if label_test.size() != torch.Size([1]):
        label_test = label_test.unsqueeze(dim=0)

    assert label_train.size() == torch.Size([1]) and label_test.size() == torch.Size([1]), "size of label should be 1"


    # calculate gradient
    ptime = time.time()
    logits_train = net(z_train.to(device))
    loss_train = loss_function(logits_train, label_train.to(device))
    grad_train = grad(loss_train, net.parameters())
    grad_train = [g for g in grad_train]
    logits_test = net(z_test.to(device))
    loss_test = loss_function(logits_test, label_test.to(device))
    grad_test = grad(loss_test, net.parameters())
    grad_test = [g for g in grad_test]
    time_gradient = time.time() - ptime


    # get tracin score
    ptime = time.time()
    score = get_tracin(grad_train, grad_test)
    time_tracin = time.time() - ptime

    total_time = time_gradient + time_tracin

    print("{:.2f}% of time used calculating gradient".format(time_gradient / total_time * 100))
    print("{:.2f}% of time used calculating inner product".format(time_tracin / total_time * 100))

    return score


def tracin_multi_single(net, zs_train, z_test, labels_train, label_test, loss=nn.CrossEntropyLoss()):
    """
    Compute tracin scores of 1 test sample versus multiple train samples
    :param net: the corresponding model
    :param zs_train: multiple train samples, a torch tensor of shape (BS, 3, H, W)
    :param z_test: single test sample, a torch tensor of shape (3, H, W)
    :param labels_train: labels corresponding to the train samples
    :param label_test: the label corresponding to the test sample
    :param loss: loss function to calculate loss for a single sample
    :return: tracin scores, a ndarray of shape (len(zs_train),)
    """
    scores = []
    n_train = zs_train.shape[0]
    for i in range(n_train):
        score = tracin_single_single(net, zs_train[i], z_test,
                                     labels_train[i], label_test)
        scores.append(score.cpu())

    return np.array(scores)


def tracin_multi_multi(net, zs_train, zs_test, labels_train, labels_test, loss_function=nn.CrossEntropyLoss()):
    """
    Compute tracin scores of multiple samples versus multiple samples
    :param net: the corresponding model
    :param zs_train: multiple train samples, a torch tensor of shape (n_train, 3, H, W)
    :param zs_test: multiple train samples, a troch tensor of shape (n_test, 3, H, W)
    :param labels_train: labels corresponding to the train samples
    :param lebals_test: labels corresponding to the test samples
    :param loss: loss function to calculate loss for a single sample
    :return: tracin scores, a ndarray of shape (n_train, n_test)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prepare model
    net.to(device)
    net.eval()

    # prepare data
    n_train = len(zs_train)
    n_test = len(zs_test)

    scores = np.zeros((n_train, n_test))

    # calculate test gradient
    test_gradients = []
    for i in range(n_test):
        z_test = zs_test[i].unsqueeze(0)
        label_test = labels_test[i].unsqueeze(0)
        logits_test = net(z_test.to(device))
        loss_test = loss_function(logits_test, label_test.to(device))
        grad_test = grad(loss_test, net.parameters())
        grad_test = [g for g in grad_test]
        test_gradients.append(grad_test)

    # calculate train gradient and score
    for i in range(n_train):
        z_train = zs_train[i].unsqueeze(0)
        label_train = labels_train[i].unsqueeze(0)
        logits_train = net(z_train.to(device))
        loss_train = loss_function(logits_train, label_train.to(device))
        grad_train = grad(loss_train, net.parameters())
        grad_train = [g for g in grad_train]
        for j, grad_test in enumerate(test_gradients):
            score = get_tracin(grad_train, grad_test)
            scores[i, j] = score

    return scores


def tracin_self(net, zs_train, labels_train, loss_function=nn.CrossEntropyLoss()):
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

    # prepare model
    net.to(device)
    net.eval()

    # prepare data
    n_train = len(zs_train)

    scores = np.zeros(n_train)

    # calculate self-influence
    for i in range(n_train):
        z_train = zs_train[i].unsqueeze(0)
        label_train = labels_train[i].unsqueeze(0)
        logits_train = net(z_train.to(device))
        loss_train = loss_function(logits_train, label_train.to(device))
        grad_train = grad(loss_train, net.parameters())
        grad_train = [g for g in grad_train]
        score = get_tracin(grad_train, grad_train)
        scores[i] = score

    return scores


def select_validation(net, valloader, method="V1"):
    """
    Select validation samples based on their accuracy on 3 checkpoints
    :param net: the model to load checkpoints
    :param valloader: whole validation data
    :return: selected validation samples
    """
    if os.path.exists("validation_correct_count.npz"):
        file = np.load("validation_correct_count.npz", allow_pickle=True)
        correct_count = file["count"]
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        path1 = "model_weights/CNN_CIFAR10_epoch1.pth"
        path2 = "model_weights/CNN_CIFAR10_epoch2.pth"
        path3 = "model_weights/CNN_CIFAR10_epoch3.pth"

        correct_wrong = []
        with torch.no_grad():
            # epoch 1
            net.load_state_dict(torch.load(path1, map_location=device))
            net.eval()
            correct_wrong_epoch = []
            for X_batch, Y_batch in valloader:
                logits = net(X_batch.to(device))
                y_pred = torch.argmax(logits, dim=1).cpu()
                correct_wrong_epoch.append(y_pred == Y_batch)
            correct_wrong_epoch = torch.hstack(correct_wrong_epoch)
            correct_wrong.append(correct_wrong_epoch)

            # epoch 2
            net.load_state_dict(torch.load(path2, map_location=device))
            net.eval()
            correct_wrong_epoch = []
            for X_batch, Y_batch in valloader:
                logits = net(X_batch.to(device))
                y_pred = torch.argmax(logits, dim=1).cpu()
                correct_wrong_epoch.append(y_pred == Y_batch)
            correct_wrong_epoch = torch.hstack(correct_wrong_epoch)
            correct_wrong.append(correct_wrong_epoch)

            # epoch 3
            net.load_state_dict(torch.load(path3, map_location=device))
            net.eval()
            correct_wrong_epoch = []
            for X_batch, Y_batch in valloader:
                logits = net(X_batch.to(device))
                y_pred = torch.argmax(logits, dim=1).cpu()
                correct_wrong_epoch.append(y_pred == Y_batch)
            correct_wrong_epoch = torch.hstack(correct_wrong_epoch)
            correct_wrong.append(correct_wrong_epoch)

            correct_wrong = torch.vstack(correct_wrong)
            correct_count = torch.sum(correct_wrong, dim=0)   # a tensor of length 10000

            np.savez("validation_correct_count.npz", count=correct_count)

    X_val_all = valloader.dataset.Data
    Y_val_all = valloader.dataset.Label

    if method == "V1":
        X_selected = []
        Y_selected = []
        for i in range(10):
            X_this_class = X_val_all[(correct_count == 3) & (Y_val_all == i).numpy()]
            Y_this_class = Y_val_all[(correct_count == 3) & (Y_val_all == i).numpy()]
            n = len(Y_this_class)
            index = np.random.permutation(n)[:2]
            X_selected.append(X_this_class[index])
            Y_selected.append(Y_this_class[index])
        X_selected = torch.cat(X_selected, dim=0)
        Y_selected = torch.cat(Y_selected, dim=0)
        return {"X": X_selected, "Y":Y_selected}
    elif method == "V2":
        X_selected = []
        Y_selected = []
        for i in range(10):
            X_this_class = X_val_all[(correct_count == 0) & (Y_val_all == i).numpy()]
            Y_this_class = Y_val_all[(correct_count == 0) & (Y_val_all == i).numpy()]
            n = len(Y_this_class)
            index = np.random.permutation(n)[:2]
            X_selected.append(X_this_class[index])
            Y_selected.append(Y_this_class[index])
        X_selected = torch.cat(X_selected, dim=0)
        Y_selected = torch.cat(Y_selected, dim=0)
        return {"X": X_selected, "Y": Y_selected}


def select_train(net, trainloader, selected_val, method="T1"):
    # prepare model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # prepare data
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label
    n_train = len(Y_train)

    selected_val_X_all = selected_val["X"]
    selected_val_Y_all = selected_val["Y"]
    n_test = len(selected_val_Y_all)

    # compute TracIn
    # TracIn_all = np.load("datasets/selected_cifar10/TracIn_all.npz", allow_pickle=True)["TracIn_all"]
    TracIn_all = np.zeros((n_train, n_test))
    for i in range(1, 4):
        # load weight
        net.load_state_dict(torch.load("model_weights/CNN_CIFAR10_epoch{}.pth".format(i), map_location=device))
        TracIn_all = TracIn_all + tracin_multi_multi(net, X_train, selected_val_X_all, Y_train, selected_val_Y_all)
    print("TracIn calculation completed")
    # net.load_state_dict(torch.load("model_weights/CNN_CIFAR10_epoch3.pth", map_location=device))
    # TracIn_all = tracin_multi_multi(net, X_train, selected_val_X_all, Y_train, selected_val_Y_all)

    if method == "T1":
        TracIn_mean = np.mean(TracIn_all, axis=1)
        lower = np.quantile(TracIn_mean, q=0.75)
        upper = np.quantile(TracIn_mean, q=1)
        index = (TracIn_mean >= lower) & (TracIn_mean <= upper) & (TracIn_mean > 0)
        X_selected = X_train[index]
        Y_selected = Y_train[index]
        return {"X":X_selected, "Y":Y_selected}
    elif method == "T2":
        TracIn_mean = np.mean(TracIn_all, axis=1)
        lower = np.quantile(TracIn_mean, q=0)
        upper = np.quantile(TracIn_mean, q=0.25)
        index = (TracIn_mean >= lower) & (TracIn_mean <= upper) & (TracIn_mean < 0)
        X_selected = X_train[index]
        Y_selected = Y_train[index]
        return {"X":X_selected, "Y":Y_selected}
    elif method == "T3":
        index = np.arange(n_train)
        for i in range(n_test):
            index_tmp = (TracIn_all[:, i] > 0)
            index = np.intersect1d(index, index_tmp)
        X_selected = X_train[index]
        Y_selected = Y_train[index]
        return {"X":X_selected, "Y":Y_selected}
    elif method == "T4":
        index = np.arange(n_train)
        for i in range(n_test):
            index_tmp = (TracIn_all[:, i] < 0)
            indx = np.intersect1d(index, index_tmp)
        X_selected = X_train[index]
        Y_selected = Y_train[index]
        return {"X": X_selected, "Y": Y_selected}
    elif method == "T5":
        TracIn_mean = np.mean(TracIn_all, axis=1)
        index = (TracIn_mean > 0)
        X_selected = X_train[index]
        Y_selected = Y_train[index]
        return {"X": X_selected, "Y": Y_selected}
    elif method == "T6":
        TracIn_mean = np.mean(TracIn_all, axis=1)
        index = (TracIn_mean < 0)
        X_selected = X_train[index]
        Y_selected = Y_train[index]
        return {"X": X_selected, "Y": Y_selected}


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prepare model and load weight
    net = CNN_CIFAR10()
    net.to(device)
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10.pth", map_location=device))

    ptime = time.time()
    # prepare data
    trainloader, valloader, testloader = prepare_CIFAR10(img_size=32)
    selected_val = select_validation(net, valloader, method="V1")
    selected_train = select_train(net, trainloader, selected_val, method="T1")

    ctime = time.time()
    print(ctime - ptime)


    stop = None



