import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

from model import CNN_CIFAR10
from utils import device, prepare_CIFAR10


def select_validation():
    trainloader, valloader, testloader = prepare_CIFAR10()
    keep_per_class = 20   # number of selected samples from each class
    X_val = valloader.dataset.Data
    Y_val = valloader.dataset.Label

    X_val_keep = []
    Y_val_keep = []
    np.random.seed(42)
    for cls in range(10):
        X_class = X_val[Y_val == cls]
        Y_class = Y_val[Y_val == cls]
        n_class = len(Y_class)
        keep_index = np.random.permutation(n_class)[:keep_per_class]
        X_val_keep.append(X_class[keep_index])
        Y_val_keep.append(Y_class[keep_index])
    X_val_keep = np.concatenate(X_val_keep, axis=0)
    Y_val_keep = np.concatenate(Y_val_keep, axis=0)
    np.savez("data/SV_random.npz", X=X_val_keep, Y=Y_val_keep)


def get_tracin(a, b):
    assert len(a) == len(b), "2 lists of gradient must have the same length"
    return sum([torch.dot(at.flatten(), bt.flatten()) for at, bt in zip(a, b)])


def save_TracIn(ckpts=np.arange(150)):
    """
    Compute TracIn score at each checkpoint
    """
    net = CNN_CIFAR10().to(device)
    net.eval()
    loss_function = nn.CrossEntropyLoss()

    # prepare data
    trainloader, valloader, testloader = prepare_CIFAR10(mode="tvt")
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label
    selected_val = np.load("data/SV_random.npz", allow_pickle=True)
    X_sv = torch.Tensor(selected_val["X"])
    Y_sv = torch.LongTensor(selected_val["Y"])

    # record learning rates
    lrs = []
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 150)
    for i in range(150):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    for epoch in ckpts:
        net.load_state_dict(torch.load(f"data/CNN_CIFAR10_epoch{epoch}.pth", map_location=device))
        net.eval()
        lr = lrs[epoch-1]

        TracIn_by_cls = {}
        for cls in range(10):
            X_train_cls = X_train[Y_train == cls]
            Y_train_cls = Y_train[Y_train == cls]
            X_sv_cls = X_sv[Y_sv == cls]
            Y_sv_cls = Y_sv[Y_sv == cls]
            n_train = len(X_train_cls)
            n_sv = len(X_sv_cls)
            TracIn_this_cls = np.zeros((n_train, n_sv))

            # calculate test gradient
            test_gradients = []
            logits = net(X_sv_cls.to(device))
            for i in range(n_sv):
                loss = loss_function(logits[i].unsqueeze(0), Y_sv_cls[i].unsqueeze(0).to(device))
                grad_tmp = grad(loss, net.parameters(), retain_graph=True)
                grad_tmp = [g for g in grad_tmp]
                test_gradients.append(grad_tmp)

            # calculate train gradient and score
            for i in range(n_train):
                logit = net(X_train_cls[i].unsqueeze(0).to(device))
                loss = loss_function(logit, Y_train_cls[i].unsqueeze(0).to(device))
                grad_train = grad(loss, net.parameters())
                grad_train = [g for g in grad_train]
                for j, grad_test in enumerate(test_gradients):
                    score = get_tracin(grad_train, grad_test)
                    TracIn_this_cls[i, j] = score * lr
            TracIn_by_cls[cls] = TracIn_this_cls

        print("Epoch {} finished".format(epoch))
        np.save(f"data/TracIn_by_class/TracIn_epoch{epoch}.npy", TracIn_by_cls)


if __name__ == "__main__":
    select_validation()