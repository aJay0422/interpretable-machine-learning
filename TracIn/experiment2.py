import torch
import torch.nn as nn
import os

from model import CNN_CIFAR10
from data import prepare_CIFAR10
from train import train_CNN_CIFAR10

if __name__ == "__main__":
    experiment_path = "./experiment2"
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    # prepare data
    trainloader, testloader = prepare_CIFAR10(mode="tt")

    # train a baseline model
    epochs = 50
    save_path = experiment_path + "/CNN_CIFAR10_basline.pth"
    train_CNN_CIFAR10(epochs, trainloader, testloader, save_path=save_path)

    # experiment2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = CNN_CIFAR10()
    net.to(device)
    epochs = 50
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(1, epochs+1):
        net.train()
        for X_batch, Y_batch in trainloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # forward
            logits = net(X_batch)
            loss = loss_function(logits, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate




