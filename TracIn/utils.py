import torch
import numpy as np
from torch.utils.data import Dataset

# from influence_functions import tracin_multi_multi, tracin_multi_single


def get_loss_acc(model, dataloader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    correct = 0
    total = 0
    total_loss = 0
    num_batches = 0
    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        total += len(Y_batch)
        num_batches += 1
        outputs = model(X_batch)
        y_pred = torch.argmax(outputs, dim=1)
        correct += torch.sum(y_pred == Y_batch).cpu().numpy()
        loss = criterion(outputs, Y_batch)
        total_loss += loss.item()
    acc = correct / total
    total_loss = total_loss / num_batches

    return total_loss, acc


class mydataset(Dataset):
    def __init__(self, X, Y):
        self.Data = torch.Tensor(X)
        self.Label = torch.LongTensor(Y)

    def __getitem__(self, item):
        return self.Data[item], self.Label[item]

    def __len__(self):
        return len(self.Label)


def get_tracin(a, b):
    """
    get tracin score by 2 lists of gradients
    :param a: a List of gradient
    :param b: a List of gradient
    :return: inner product of 2 gradient
    """
    assert len(a) == len(b), "2 list of gradient must have the same length"
    return sum([torch.dot(at.flatten(), bt.flatten()) for at, bt in zip(a, b)])


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
        selected_idx = np.random.permutation(tmp_n)[:5]
        selected_samples[i] = (tmp_X[selected_idx], tmp_Y[selected_idx])

    return selected_samples


def select_train(net, trainloader, selected_val):
    X_train = trainloader.dataset.Data
    Y_train = trainloader.dataset.Label
    n_train = len(Y_train)

    selected_train = []

    # level 1 validation samples
    X_val, Y_val = selected_val[0]
    for i, z_val in enumerate(X_val):
        score = tracin_multi_single(net, X_train, z_val, Y_train, Y_val[i])
        lower = torch.quantile(score, 0.75)
        upper = torch.quantile(score, 1)
        selected_idx = torch.arange(n_train)[lower <= score <= upper]
        selected_train.append(selected_idx)

    return torch.unique(torch.hstack(selected_train))



