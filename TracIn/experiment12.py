"""
Replace 3-layer CNN with resnet
"""
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import models
import numpy as np
import time
from tqdm import tqdm

from data import prepare_CIFAR10
from utils import device, get_loss_acc, get_acc_by_cls


def train_resnet18(epochs,
                   trainloader, testloader,
                   seed=20220906,
                   save_path=None,
                   epoch_path=None):
    # set up
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

    # prepare pretrained model
    net = models.resnet18(pretrained=False)

    # # freeze all layers
    # for param in net.parameters():
    #     param.requires_grad = False

    # replce the classification head
    net.fc = nn.Linear(512, 10)
    net.to(device)

    # train model
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-30)
    loss_function = nn.CrossEntropyLoss()
    print("Trained on {}".format(device))
    if trainloader.dataset.indices:
        print("Trained on {} samples".format(len(trainloader.dataset.indices)))
    else:
        print("Trained on {} samples".format(len(trainloader.dataset.targets)))

    # initial
    with torch.no_grad():
        net.eval()
        # evaluate train
        train_loss, train_acc = get_loss_acc(net, trainloader, loss_function)
        # evaluate test
        test_loss, test_acc = get_loss_acc(net, testloader, loss_function)

    print("Epoch{}/{} train loss: {} test loss: {} train acc: {} test acc: {}".format(0, epochs,
                                                                                          train_loss, test_loss,
                                                                                          train_acc, test_acc))
    best_test_acc = 0
    for epoch in range(epochs):
        net.train()
        net.to(device)
        ptime = time.time()
        for X_batch, Y_batch in tqdm(trainloader):
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
            # evaluate train
            train_loss, train_acc = get_loss_acc(net, trainloader, loss_function)
            # evaluate test
            test_loss, test_acc = get_loss_acc(net, testloader, loss_function)

        # record train time
        ctime = time.time()

        print("Epoch{}/{}  train loss: {}  test loss: {}  train acc: {}  test acc: {}  time: {}s".format(epoch + 1,
                                                                                                         epochs,
                                                                                                         train_loss,
                                                                                                         test_loss,
                                                                                                         train_acc,
                                                                                                         test_acc,
                                                                                                         int(ctime - ptime)))

        # save model weights if it is the best
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(net.state_dict(), save_path)
            print("Saved")

        if epoch_path:
            torch.save(net.state_dict(), epoch_path.format(epoch+1))


def experiment_random_drop(keep_ratios, seed=42):
    for keep_ratio in keep_ratios:
        trainloader, valloader, testloader = prepare_CIFAR10()
        train_set = trainloader.dataset
        n_train = len(train_set.indices)
        n_keep = int(n_train * keep_ratio)
        torch.manual_seed(seed)
        train_set_keep, train_set_drop = random_split(train_set, [n_keep, n_train-n_keep])
        trainloader_keep = DataLoader(train_set_keep, batch_size=128,
                                      shuffle=True)
        save_path = "experiment12/resnet18_randomkeep/ResNet18_rndkeep_{}.pth".format(keep_ratio)
        train_resnet18(230, trainloader_keep, valloader, save_path=save_path)


def evaluate_random_drop(keep_ratios=[0.1, 0.2, 0.4, 0.6, 0.8]):
    accs_all = {}
    trainloader, valloader, testloader = prepare_CIFAR10()

    # get baseline acc
    net = models.resnet18(pretrained=False)
    net.fc = nn.Linear(512, 10)
    net.load_state_dict(torch.load("experiment12/resnet18/ResNet18.pth", map_location=device))
    _, baseline_acc = get_loss_acc(net, testloader)

    # get random keep accuracy
    accs = np.zeros(len(keep_ratios) + 1)
    accs[-1] = baseline_acc
    for i, keep_ratio in enumerate(keep_ratios):
        net.load_state_dict(
            torch.load(f"experiment12/resnet18_randomkeep/ResNet18_rndkeep_{keep_ratio}.pth", map_location=device)
        )
        _, acc = get_loss_acc(net, testloader)
        accs[i] = acc
    accs_all["random"] = accs

    # plot curve
    fig = plt.figure(figsize=(10, 10))
    for name in accs_all.keys():
        plt.plot(keep_ratios+[1], accs_all[name], ".-", label=name)
        print(name)
        print(accs_all[name])
    plt.legend()
    plt.show()




if __name__ == "__main__":
    evaluate_random_drop()