import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from data import prepare_CIFAR10
from model import resnet34, CNN_CIFAR10, resnet50
from utils import get_loss_acc, mydataset, device


def tune_resnet34():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prepare dataloaders
    trainloader, testloader = prepare_CIFAR10(224)

    # prepare model
    net = resnet34()
    net.to(device)

    # load pretrained weights
    pretrained_weight_path = "model_weights/resnet34-333f7ec4.pth"
    net.load_state_dict(torch.load(pretrained_weight_path, map_location=device))

    # freeze all layers
    for param in net.parameters():
        param.requires_grad = False

    # replace the fc layer
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 10)

    # fine tune model
    epochs = 50
    optimizer = torch.optim.SGD(net.parameters(), lr=0.00001)
    loss_function = nn.CrossEntropyLoss()
    save_path = "model_weights/resnet34_cifar10.pth"
    print("Trained on {}".format(device))
    net.to(device)

    best_test_acc = 0
    net.train()
    for epoch in range(50):
        for X_batch, Y_batch in trainloader:
            X_batch.to(device)
            Y_batch.to(device)
            # forward
            logits = net(X_batch)
            loss = loss_function(logits, Y_batch)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        with torch.no_grad():
            net.eval()
            # evaluate train
            train_loss, train_acc = get_loss_acc(net, trainloader, nn.CrossEntropyLoss())
            # evaluate test
            test_loss, test_acc = get_loss_acc(net, testloader, nn.CrossEntropyLoss())

        print("Epoch{}/{}  train loss: {}  test loss: {}  train acc: {}  test acc: {}".format(epoch+1, epochs,
                                                                                              train_loss, test_loss,
                                                                                              train_acc, test_acc))

        # save model weights if it's the best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(net.state_dict(), save_path)
            print("Saved")


def train_CNN_CIFAR10(epochs,
                      trainloader,
                      testloader,
                      seed=20220718, get_summary=False,
                      load_path=None, save_path="model_weights/CNN_CIFAR10.pth",
                      epoch_path=None):
    # setup
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if get_summary:
        summary = {"train loss":[],
                   "test loss":[],
                   "train acc":[],
                   "test acc":[]}

    # prepare model
    net = CNN_CIFAR10(num_classes=10)
    net.to(device)
    if load_path is not None:
        net.load_state_dict(torch.load(load_path, map_location=device))

    # train model
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    print("Trained on {}, {} samples".format(device, len(trainloader.dataset.Label)))
    net.to(device)

    # initial
    with torch.no_grad():
        net.eval()
        # evaluate train
        train_loss, train_acc = get_loss_acc(net, trainloader, loss_function)
        # evaluate test
        test_loss, test_acc = get_loss_acc(net, testloader, loss_function)



    best_test_acc = 0
    for epoch in range(epochs):
        net.train()
        net.to(device)
        for X_batch, Y_batch in trainloader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            # forward
            logits = net(X_batch)
            loss = loss_function(logits, Y_batch)
            # backward and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        with torch.no_grad():
            net.eval()
            # evaluate train
            train_loss, train_acc = get_loss_acc(net, trainloader, loss_function)
            # evaluate test
            test_loss, test_acc = get_loss_acc(net, testloader, loss_function)

        print("Epoch{}/{}  train loss: {}  test loss: {}  train acc: {}  test acc: {}".format(epoch + 1, epochs,
                                                                                              train_loss, test_loss,
                                                                                              train_acc, test_acc))
        if get_summary:
            summary["train loss"].append(train_loss)
            summary["test loss"].append(test_loss)
            summary["train acc"].append(train_acc)
            summary["test acc"].append(test_acc)

        # save model weights of it's the best
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(net.state_dict(), save_path)
            print("Saved")

        if epoch_path:
            torch.save(net.state_dict(),
                       epoch_path.format(epoch+1))

    if get_summary:
        return summary


def train_Resnet_CIFAR10(epochs,
                         trainloader,
                         testloader,
                         seed=20220718, get_summary=False,
                         load_path=None, save_path="model_weights(Resnet50)/Resnet_CIFAR10.pth",
                         epoch_path=None):
    # set up
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if get_summary:
        summary = {"train loss":[],
                   "test loss":[],
                   "train acc":[],
                   "test acc":[]}

    # prepare model
    net = resnet50(num_classes=10)
    net.to(device)
    if load_path is not None:
        net.load_state_dict(torch.load(load_path, map_location=device))

    # train model
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    loss_function = nn.CrossEntropyLoss()
    print("Trained on {}".format(device))

    # initial
    with torch.no_grad():
        net.eval()
        # evaluate train
        train_loss, train_acc = get_loss_acc(net, trainloader, loss_function)
        # evaluate test
        test_loss, test_acc = get_loss_acc(net, testloader, loss_function)

    print("Epoch{}/{}  train loss: {}  test loss: {}  train acc: {}  test acc: {}".format(0, epochs,
                                                                                          train_loss, test_loss,
                                                                                          train_acc, test_acc))

    best_test_acc = 0
    for epoch in range(epochs):
        net.train()
        net.to(device)
        ptime = time.time()
        for X_batch, Y_batch in tqdm(trainloader):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # forward
            logits = net(X_batch)
            loss = loss_function(logits, Y_batch)

            #backward and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        with torch.no_grad():
            net.eval()
            # evaluate train
            train_loss, train_acc = get_loss_acc(net, trainloader, loss_function)
            # evaluate test
            test_loss, test_acc = get_loss_acc(net, testloader, loss_function)

        # record train time
        ctime = time.time()

        print("Epoch{}/{}  train loss: {}  test loss: {}  train acc: {}  test acc: {}  time: {}s".format(epoch + 1, epochs,
                                                                                              train_loss, test_loss,
                                                                                              train_acc, test_acc,
                                                                                              int(ctime-ptime)))

        if get_summary:
            summary["train loss"].append(train_loss)
            summary["test loss"].append(test_loss)
            summary["train acc"].append(train_acc)
            summary["test acc"].append(test_acc)

        # save model weights if it is the best
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(net.state_dict(), save_path)
            print("Saved")

        scheduler.step()

        if epoch_path:
            torch.save(net.state_dict(),
                       epoch_path.format(epoch+1))

    if get_summary:
        return summary


if __name__ == "__main__":
    trainloader, valloader, testloader = prepare_CIFAR10()
    save_path = "experiment10/model_weights_70epochs/CNN_CIFAR10.pth"
    epoch_path = "experiment10/model_weights_70epochs/CNN_CIFAR10_epoch{}.pth"
    train_CNN_CIFAR10(100, trainloader, valloader, seed=20220830, save_path=save_path, epoch_path=epoch_path)
