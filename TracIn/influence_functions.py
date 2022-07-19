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
        scores.append(score)

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



if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prepare model and load weight
    net = CNN_CIFAR10()
    net.to(device)
    net.load_state_dict(torch.load("model_weights/CNN_CIFAR10.pth", map_location=device))

    # prepare data
    trainloader, testloader = prepare_CIFAR10(img_size=32)
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

