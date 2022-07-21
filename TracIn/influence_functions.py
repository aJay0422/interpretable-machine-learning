import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np
import time

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
    logits_train = net(z_train.to(device))
    loss_train = loss_function(logits_train, label_train.to(device))
    grad_train = grad(loss_train, net.parameters())
    grad_train = [g for g in grad_train]
    logits_test = net(z_test.to(device))
    loss_test = loss_function(logits_test, label_test.to(device))
    grad_test = grad(loss_test, net.parameters())
    grad_test = [g for g in grad_test]

    # get tracin score
    score = get_tracin(grad_train, grad_test)

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


def tracin_multi_multi(net, zs_train, zs_test, labels_train, labels_test, loss=nn.CrossEntropyLoss()):
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
    n_train = zs_train.shape[0]
    n_test = zs_test.shape[0]
    scores = np.zeros((n_train, n_test))
    for i in range(n_test):
        img_test = zs_test[i]
        label_test = labels_test[i]
        this_scores = tracin_multi_single(net, zs_train, img_test,
                                          labels_train, label_test)
        scores[:, i] = this_scores

    return scores


def select_validation(net, valloader):
    """
    Select validation samples based on their accuracy on 3 checkpoints
    :param net: the model to load checkpoints
    :param valloader: whole validation data
    :return: selected validation samples
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    path1 = "model_weights/CNN_CIFAR10_epoch1.pth"
    path2 = "model_weights/CNN_CIFAR10_epoch2.pth"
    path3 = "model_weights/CNN_CIFAR10_epoch3.pth"

    correct_wrong = []

    # epoch1
    net.load_state_dict(torch.load(path1, map_location=device))
    correct_wrong_epoch = []
    for X_batch, Y_batch in valloader:
        logits = net(X_batch)
        y_pred = torch.argmax(logits, dim=1)
        correct_wrong_epoch.append(y_pred == Y_batch)
    correct_wrong_epoch = torch.hstack(correct_wrong_epoch)
    correct_wrong.append(correct_wrong_epoch)

    # epoch2
    net.load_state_dict(torch.load(path2, map_location=device))
    correct_wrong_epoch = []
    for X_batch, Y_batch in valloader:
        logits = net(X_batch)
        y_pred = torch.argmax(logits, dim=1)
        correct_wrong_epoch.append(y_pred == Y_batch)
    correct_wrong_epoch = torch.hstack(correct_wrong_epoch)
    correct_wrong.append(correct_wrong_epoch)

    # epoch3
    net.load_state_dict(torch.load(path3, map_location=device))
    correct_wrong_epoch = []
    for X_batch, Y_batch in valloader:
        logits = net(X_batch)
        y_pred = torch.argmax(logits, dim=1)
        correct_wrong_epoch.append(y_pred == Y_batch)
    correct_wrong_epoch = torch.hstack(correct_wrong_epoch)
    correct_wrong.append(correct_wrong_epoch)

    correct_wrong = torch.vstack(correct_wrong)
    correct_count = torch.sum(correct_wrong, dim=0)   # a tensor of length 10000

    selected_samples = {}
    for i in range(4):
        tmp_X = valloader.dataset.Data[correct_count == i]
        tmp_Y = valloader.dataset.Label[correct_count == i]
        tmp_n = len(tmp_Y)
        selected_idx = np.random.permutation(tmp_n)[:2]
        selected_samples[i] = (tmp_X[selected_idx], tmp_Y[selected_idx])

    np.savez("datasets/selected_cifar10/selected_val.npz", selected_val=selected_samples)
    return selected_samples


def select_train(net, trainloader, selected_val):
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label
    n_train = len(Y_train)

    selected_train = []

    # level 1 validation samples (hardest samples, 0 correct in the 3 epochs)
    X_val, Y_val = selected_val[0]
    selected_train1 = np.arange(n_train)
    for i, z_val in enumerate(X_val):
        score = tracin_multi_single(net, X_train, z_val, Y_train, Y_val[i])
        lower = np.quantile(score, 0)
        upper = np.quantile(score, 0.25)
        selected_idx = np.arange(n_train)[(lower <= score) * (score <= upper)]
        selected_train1 = np.intersect1d(selected_train1, selected_idx)
    selected_train.append(selected_train1)
    print(len(selected_train1))

    # level 2 validation samples
    X_val, Y_val = selected_val[1]
    selected_train2 = np.arange(n_train)
    for i, z_val in enumerate(X_val):
        score = tracin_multi_single(net, X_train, z_val, Y_train, Y_val[i])
        lower = np.quantile(score, 0.25)
        upper = np.quantile(score, 0.5)
        selected_idx = np.arange(n_train)[(lower <= score) * (score <= upper)]
        selected_train2 = np.intersect1d(selected_train2, selected_idx)
    selected_train.append(selected_train2)
    print(len(selected_train2))

    # level 3 validation samples
    X_val, Y_val = selected_val[2]
    selected_train3 = np.arange(n_train)
    for i, z_val in enumerate(X_val):
        score = tracin_multi_single(net, X_train, z_val, Y_train, Y_val[i])
        lower = np.quantile(score, 0.5)
        upper = np.quantile(score, 0.75)
        selected_idx = np.arange(n_train)[(lower <= score) * (score <= upper)]
        selected_train3 = np.intersect1d(selected_train3, selected_idx)
    selected_train.append(selected_train3)
    print(len(selected_train3))

    # level 4 validation samples (easiest samples, 3 correct in the 3 epochs)
    X_val, Y_val = selected_val[3]
    selected_train4 = np.arange(n_train)
    for i, z_val in enumerate(X_val):
        score = tracin_multi_single(net, X_train, z_val, Y_train, Y_val[i])
        lower = np.quantile(score, 0.75)
        upper = np.quantile(score, 1)
        selected_idx = np.arange(n_train)[(lower <= score) * (score <= upper)]
        selected_train4 = np.intersect1d(selected_train4, selected_idx)
    selected_train.append(selected_train4)
    print(len(selected_train4))

    selected_train_idx = np.unique(np.hstack(selected_train))

    selected_X_train = X_train[selected_train_idx]
    selected_Y_train = Y_train[selected_train_idx]
    np.savez("datasets/selected_cifar10/selected_train.npz", X=selected_X_train, Y=selected_Y_train)
    return selected_X_train, selected_Y_train


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prepare model and load weight
    net = CNN_CIFAR10()
    net.to(device)
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10.pth", map_location=device))

    # prepare data
    trainloader, valloader, testloader = prepare_CIFAR10(img_size=32)
    img_train = trainloader.dataset.Data[0:100] # samples from train set
    label_train = trainloader.dataset.Label[0:100]
    img_test = testloader.dataset.Data[0:20]   # a single sample from test set
    label_test = testloader.dataset.Label[0:20]

    print("Started")
    ptime = time.time()
    scores = tracin_multi_multi(net, img_train, img_test, label_train, label_test)
    ctime = time.time()
    n = scores.shape[0] * scores.shape[1]
    print("Infer time: {}({} seconds every 10000 infers)".format(int(ctime - ptime),
        int((ctime - ptime) / n * 10000)))

