import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils import device, get_loss_acc, prepare_CIFAR10
from model import CNN_CIFAR10

def train_CNN_CIFAR10(epochs,
                      trainloader,
                      testloader,
                      seed=20220718,
                      load_path=None,
                      save_path=None,
                      epoch_path=None):
    # Setup
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Prepare model
    net = CNN_CIFAR10(num_classes=10)
    net.to(device)
    if load_path is not None:
        net.load_state_dict(torch.load(load_path, map_location=device))

    # train model
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print("Trained on {}, {} samples".format(device,
                                             len(trainloader.dataset.Label)))
    net.to(device)

    # Initial state
    with torch.no_grad():
        net.eval()
        train_loss, train_acc = get_loss_acc(net, trainloader)
        test_loss, test_acc = get_loss_acc(net, testloader)

    best_test_acc = 0
    for epoch in range(epochs):
        net.train()
        net.to(device)
        for X_batch, Y_batch in trainloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            # forward
            logits = net(X_batch)
            loss = loss_function(logits, Y_batch)
            # backward and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # evaluate
        with torch.no_grad():
            net.eval()
            train_loss, train_acc = get_loss_acc(net, trainloader)
            test_loss, test_acc  =get_loss_acc(net, testloader)

        print("Epoch{}/{}  train loss: {}  test loss: {}  train acc: {}  test acc: {}".format(epoch + 1, epochs,
                                                                                              train_loss, test_loss,
                                                                                              train_acc, test_acc))

        # Save model weights if it's the best
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(net.state_dict(), save_path)
            print("Saved")

        # Save checkpoints at each epoch
        if epoch_path:
            torch.save(net.state_dict(), epoch_path.format(epoch + 1))


def main():
    trainloader, valloader, testloader = prepare_CIFAR10()
    save_path = "data/CNN_CIFAR10_full/CNN_CIFAR10.pth"
    epoch_path = "data/CNN_CIFAR10_full/CNN_CIFAR10_epoch{}.pth"
    train_CNN_CIFAR10(150, trainloader, valloader, save_path=save_path, epoch_path=epoch_path)


def training_curve():
    net = CNN_CIFAR10().to(device)
    trainloader, valloader, testloader = prepare_CIFAR10()
    weight_path = "data/CNN_CIFAR10_full/CNN_CIFAR10_epoch{}.pth"
    losses = []
    for epoch in range(1, 151):
        net.load_state_dict(torch.load(weight_path.format(epoch), map_location=device))
        loss, acc = get_loss_acc(net, trainloader)
        losses.append(loss)
    plt.plot(np.arange(1, 151), losses, ".-", label="train loss")
    plt.show()


if __name__ == "__main__":
    training_curve()